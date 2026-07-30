import torch

class RopeEmbedding(torch.nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        i = torch.arange(0, head_dim, 2).float()
        theta = 1.0 / (base ** (i / head_dim))          # (head_dim/2,)
        positions = torch.arange(max_seq_len).float()    # (max_seq_len,)
        angles = torch.outer(positions, theta)            # (max_seq_len, head_dim/2)
        angles = angles.repeat_interleave(2, dim=-1)      # (max_seq_len, head_dim)

        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, x):
        B, n_heads, T, head_dim = x.shape

        cos = self.cos[:T]   # (T, head_dim)
        sin = self.sin[:T]   # (T, head_dim)

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        x_rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

        return x * cos + x_rotated * sin