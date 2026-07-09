import torch

class Expert(torch.nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4*embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4*embed_dim, embed_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)