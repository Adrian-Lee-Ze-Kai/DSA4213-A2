import numpy as np, torch
from torch.utils.data import Dataset

class CharLMSeqDataset(Dataset):
    def __init__(self, txt_path, block_size=128, chars=None, stoi=None, itos=None):
        with open(txt_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        if chars is None:
            self.chars = sorted(list(set(self.text)))
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.chars = chars
            self.stoi = stoi
            self.itos = itos
        self.data = np.array([self.stoi.get(ch, 0) for ch in self.text], dtype=np.int64)
        self.block = block_size

    def __len__(self):
        return len(self.data) - self.block - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx+self.block])
        y = torch.from_numpy(self.data[idx+1 : idx+self.block+1])
        return x, y