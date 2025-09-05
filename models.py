import torch

class LSTMLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size=256, emb_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
        self.fc.weight = self.emb.weight 

    def forward(self, x, targets=None):
        x = self.emb(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
