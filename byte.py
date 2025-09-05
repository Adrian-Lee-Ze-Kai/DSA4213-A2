# byte_dataset.py
import numpy as np, torch
from torch.utils.data import Dataset

class ByteLMSeqDataset(Dataset):
    """
    Loads byte-level tokens from a .bin file and serves sliding windows.
    For each index i, returns:
      x = bytes[i : i+block_size]
      y = bytes[i+1 : i+block_size+1]
    """
    def __init__(self, bin_path, block_size=128):
        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.block = block_size

    def __len__(self):
        return len(self.data) - self.block - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx+self.block].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1 : idx+self.block+1].astype(np.int64))
        return x, y
