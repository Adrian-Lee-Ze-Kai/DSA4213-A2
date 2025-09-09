import numpy as np, torch
from torch.utils.data import Dataset
import re

class WordLMSeqDataset(Dataset):
    """
    Loads word-level tokens from a text file and serves sliding windows.
    For each index i, returns:
      x = words[i : i+block_size]
      y = words[i+1 : i+block_size+1]
    """
    def __init__(self, txt_path, block_size=32):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Simple word tokenizer: split on whitespace and punctuation
        self.words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        self.vocab = sorted(set(self.words))
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.data = np.array([self.stoi[w] for w in self.words], dtype=np.int64)
        self.block = block_size

    def __len__(self):
        return len(self.data) - self.block - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx+self.block])
        y = torch.from_numpy(self.data[idx+1 : idx+self.block+1])
        return x, y