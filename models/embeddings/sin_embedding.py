import torch
import math

class SinosoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        self._precompute()

    def forward(self, x):
        B, T = x.shape

        return self.embed_map[:T]

    def _precompute(self):
        positions = torch.arange(self.max_length).unsqueeze(1)
        pair_idx = torch.arange(0, self.d_model, 2)

        div_terms = torch.exp(-pair_idx * (math.log(10000)/self.d_model))

        angles = positions * div_terms

        embed_map = torch.zeros(self.max_length, self.d_model)
        embed_map[:, 0::2] = torch.sin(angles)
        embed_map[:, 1::2] = torch.cos(angles)

        self.register_buffer("embed_map", embed_map)
            
