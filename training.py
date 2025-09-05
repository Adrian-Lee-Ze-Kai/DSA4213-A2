import torch
from torch.utils.data import DataLoader
from byte import ByteLMSeqDataset
from models import LSTMLanguageModel
from math import exp

# Hyperparams
block_size = 128
batch_size = 64
epochs     = 3
device     = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
train_ds = ByteLMSeqDataset("monkeypox_ds/train.bin", block_size)
val_ds   = ByteLMSeqDataset("monkeypox_ds/val.bin",   block_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

# Model
model = LSTMLanguageModel().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

# Training loop
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Validation CE & PPL
    model.eval()
    tot_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            tot_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    ce = tot_loss / n
    ppl = exp(ce)
    print(f"Epoch {epoch+1}: val CE={ce:.4f}, PPL={ppl:.2f}")
