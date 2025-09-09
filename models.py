# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        emb_dim=128,        
        hidden_dim=256,     
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)         # post-RNN dropout
        self.fc = nn.Linear(hidden_dim, vocab_size)  # <-- NOT tied to emb

    def forward(self, x, targets=None, state=None):
        """
        x: (B, T) int64 token ids
        targets: (B, T) optional
        state: (h0, c0) optional for TBPTT (tuples of shape [num_layers, B, hidden_dim])
        """
        x = self.emb(x)                # (B, T, emb_dim)
        out, state = self.lstm(x, state)  # out: (B, T, hidden_dim)
        out = self.dropout(out)
        logits = self.fc(out)          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss
