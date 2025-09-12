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
        num_layers=2,
        dropout=0.0  # Change dropout here for LSTM!!
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


class SimpleTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        n_heads=16,
        n_layers=2,
        ff_dim=256, 
        dropout=0.0,  # Change dropout here for Transformer!!
        max_seq_len=128
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        # x: (B, T) token ids
        B, T = x.size()
        x = self.emb(x) + self.pos_emb[:, :T, :]  # (B, T, emb_dim)
        x = self.dropout(x)
        out = self.transformer(x)  # (B, T, emb_dim)
        logits = self.fc(out)      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss