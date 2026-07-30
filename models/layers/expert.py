import torch

class Expert(torch.nn.Module):
    def __init__(self, embed_dim, d_ff=None, dropout=0.1):
        super().__init__()

        # Fine-grained experts use d_ff < 4*embed_dim so that top_k of them
        # cost the same FLOPs as one dense FFN
        d_ff = d_ff if d_ff is not None else 4 * embed_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, d_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(d_ff, embed_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
