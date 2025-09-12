import os, time
import torch
import matplotlib.pyplot as plt
from math import exp
from torch.utils.data import DataLoader
from words import WordLMSeqDataset
from models import SimpleTransformerLM

# ----------------- CONFIG -----------------
block_size = 128
batch_size = 32
epochs     = 3
lr         = 3e-4
device     = "cuda" if torch.cuda.is_available() else "cpu"
data_dir   = "monkeypox_ds"
runs_dir   = "runs"; os.makedirs(runs_dir, exist_ok=True)
# ------------------------------------------

@torch.no_grad()
def sample_words(model, dataset, device, prompt="Monkeypox virus", max_new_tokens=100, temperature=1.0):
    model.eval()
    stoi, itos = dataset.stoi, dataset.itos
    import re
    prompt_tokens = re.findall(r"\w+|[^\w\s]", prompt, re.UNICODE)
    x = torch.tensor([stoi.get(w, 0) for w in prompt_tokens], dtype=torch.long, device=device)[None, :]
    generated = set()
    for _ in range(max_new_tokens):
        x_cond = x[:, -dataset.block:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.add(next_id.item())
        x = torch.cat([x, next_id], dim=1)
    out = " ".join([itos[i.item()] for i in x.squeeze(0)])
    return out

# --------------- data ---------------------
train_ds = WordLMSeqDataset("monkeypox_ds/train.txt", block_size)
voc, stoi, itos = train_ds.vocab, train_ds.stoi, train_ds.itos
val_ds = WordLMSeqDataset("monkeypox_ds/val.txt", block_size, voc, stoi, itos)
test_ds = WordLMSeqDataset("monkeypox_ds/test.txt", block_size, voc, stoi, itos)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=True)

# -------------- model/optim ---------------
vocab_size = len(train_ds.vocab)
model = SimpleTransformerLM(
    vocab_size=vocab_size,
    emb_dim=128,
    n_heads=4,
    n_layers=2,
    ff_dim=256,
    max_seq_len=block_size
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

# -------- training + epoch logging --------
train_losses, val_losses = [], []
t0 = time.time()

for epoch in range(1, epochs+1):
    # ---- train (average CE over epoch) ----
    model.train()
    sum_loss, count = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        opt.step()
        sum_loss += loss.item() * xb.size(0)
        count    += xb.size(0)
    train_ce = sum_loss / max(1, count)
    train_losses.append(train_ce)

    # ---- validate (average CE over epoch) ----
    model.eval()
    v_sum, v_cnt = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, vloss = model(xb, yb)
            v_sum += vloss.item() * xb.size(0)
            v_cnt += xb.size(0)
    val_ce = v_sum / max(1, v_cnt)
    val_losses.append(val_ce)

    print(f"Epoch {epoch}: train CE={train_ce:.4f}  val CE={val_ce:.4f}  val PPL={exp(val_ce):.2f}")

total_sec = time.time() - t0
print(f"Training time: {total_sec:.1f}s ({total_sec/60:.2f} min)")

# -------- final test evaluation --------
model.eval()
t_sum, t_cnt = 0.0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        _, tloss = model(xb, yb)
        t_sum += tloss.item() * xb.size(0)
        t_cnt += xb.size(0)
test_ce = t_sum / max(1, t_cnt)
test_ppl = exp(test_ce)
print(f"\nFinal Test CE={test_ce:.4f}, PPL={test_ppl:.2f}")

# -------- generate final samples (once) --------
for T in (0.7, 1.0, 1.3):
    txt = sample_words(model, train_ds, device, prompt="Monkeypox virus", max_new_tokens=300, temperature=T)
    print(f"\n--- Final Sample (T={T}) ---\n{txt}\n--- End sample ---\n")

# -------- plot epoch curves --------
epochs_axis = list(range(1, epochs+1))
plt.figure(figsize=(7,4))
plt.plot(epochs_axis, train_losses, marker="o", label="train CE (epoch)")
plt.plot(epochs_axis, val_losses,   marker="o", label="val CE (epoch)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy (nats)")
plt.title("Training / Validation Loss (per epoch)")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(runs_dir, "loss_curves_epoch_transformer.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved loss plot to {plot_path}")