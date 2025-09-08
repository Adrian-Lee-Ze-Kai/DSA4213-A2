import os, time
import torch
import matplotlib.pyplot as plt
from math import exp
from torch.utils.data import DataLoader
from byte import ByteLMSeqDataset
from models import LSTMLanguageModel

# ----------------- CONFIG -----------------
block_size = 256
batch_size = 64
epochs     = 1          # you can keep 1 and still get a curve
lr         = 1e-3
device     = "cuda" if torch.cuda.is_available() else "cpu"
data_dir   = "monkeypox_ds"
runs_dir   = "runs"; os.makedirs(runs_dir, exist_ok=True)
# ------------------------------------------

@torch.no_grad()
def sample_bytes(model, device, prompt="Monkeypox virus ", max_new_tokens=300, temperature=1.0):
    model.eval()
    x = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return bytes(x.squeeze(0).tolist()).decode("utf-8", errors="ignore")

def ema_smooth(values, alpha=0.98):
    """Exponential moving average for a nice smooth curve."""
    sm = []
    for v in values:
        if not sm: sm.append(v)
        else: sm.append(alpha*sm[-1] + (1-alpha)*v)
    return sm

# --------------- data ---------------------
train_ds = ByteLMSeqDataset(f"{data_dir}/train.bin", block_size)
val_ds   = ByteLMSeqDataset(f"{data_dir}/val.bin",   block_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

# -------------- model/optim ---------------
model = LSTMLanguageModel().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

# -------- training + logging --------------
train_batch_losses = []   # batch-wise (many points even in 1 epoch)
train_batch_steps  = []   # global step index for x-axis
val_losses         = []   # per-epoch

global_step = 0
t0 = time.time()

for epoch in range(1, epochs+1):
    # --- train (batch-wise logging) ---
    model.train()
    epoch_loss_sum, epoch_count = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # batch logging
        train_batch_losses.append(loss.item())
        train_batch_steps.append(global_step)
        global_step += 1

        epoch_loss_sum += loss.item() * xb.size(0)
        epoch_count    += xb.size(0)
    train_ce_epoch = epoch_loss_sum / max(1, epoch_count)

    # --- validate (per-epoch) ---
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

    print(f"Epoch {epoch}: train CE={train_ce_epoch:.4f}  val CE={val_ce:.4f}  val PPL={exp(val_ce):.2f}")

    # --- generation at different temperatures ---
    for T in (0.7, 1.0, 1.3):
        txt = sample_bytes(model, device, prompt="Monkeypox virus ", max_new_tokens=300, temperature=T)
        print(f"\n--- Sample (T={T}) ---\n{txt}\n--- End sample ---\n")

total_sec = time.time() - t0
print(f"Training time: {total_sec:.1f}s ({total_sec/60:.2f} min)")

# -------- plot curves --------
plt.figure(figsize=(7,4))
# batch-wise smoothed training curve
sm_train = ema_smooth(train_batch_losses, alpha=0.98)
plt.plot(train_batch_steps, sm_train, label="train CE (batch, EMA)", linewidth=1.5)

# epoch val points (and optional line)
if len(val_losses) > 0:
    plt.plot(
        [train_batch_steps[-1] * (i+1) / epochs for i in range(len(val_losses))],
        val_losses, "o-", label="val CE (epoch)", linewidth=1.5
    )

plt.xlabel("Training step")
plt.ylabel("Cross-Entropy (nats)")
plt.title("Training / Validation Loss")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(runs_dir, "loss_curves_batches.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved loss plot to {plot_path}")
