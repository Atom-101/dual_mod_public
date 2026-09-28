"""LSTM baseline: the classic nonlinear recurrence on the formal-language protocol (tied embedding / head)."""
import torch


class LSTMLM(torch.nn.Module):
    def __init__(self, vocab, d, L):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.lstm = torch.nn.LSTM(d, d, num_layers=L, batch_first=True)
        self.head = torch.nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return (self.head(h),)
