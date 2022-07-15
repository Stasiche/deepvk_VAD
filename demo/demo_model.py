import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModule(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first,
                 bidirectional=False):
        super(GRUModule, self).__init__()

        self.gru = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x


class ModelGRU(nn.Module):
    def __init__(self, n_rnn_layers: int, rnn_dim: int, dropout: float = 0.1) -> None:
        super(ModelGRU, self).__init__()
        gru_lst = [
            GRUModule(rnn_dim=rnn_dim,
                      hidden_size=rnn_dim, dropout=dropout, batch_first=True)
            for _ in range(n_rnn_layers)
        ]
        self.projector = nn.Linear(2*rnn_dim, rnn_dim)
        self.grus = nn.Sequential(*gru_lst)
        self.classifier = nn.Linear(rnn_dim, 2)

    @property
    def device(self):
        return next(self.classifier.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        x = self.grus(x)
        x = self.classifier(x)
        return x

