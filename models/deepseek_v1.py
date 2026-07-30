import torch
from models.blocks.rope_mha_transformer import RopeMHATransformer

class DeepSeek(torch.nn.Module):
    def __init__(self, vocab_size, d_in, max_seq_length, d_transformer, n_blocks, transformer_n_heads, num_experts=4, top_k=2, d_ff=None, dropout=0.2, gamma=0.001):
        super().__init__()

        self.d_in = d_in

        self.input_embedding = torch.nn.Embedding(vocab_size, d_in)

        self.n_blocks = n_blocks
        self.transformer_blocks = torch.nn.ModuleList([
            RopeMHATransformer(self.d_in, d_transformer, transformer_n_heads, max_seq_length, dropout,
                               num_experts=num_experts, top_k=top_k, d_ff=d_ff, gamma=gamma)
            for _ in range(n_blocks)
        ])

        self.final_ln = torch.nn.LayerNorm(d_in)

        self.out_proj = torch.nn.Linear(d_in, vocab_size)


    def forward(self, x):
        B, T = x.shape

        # Token embedding only -- position is injected by RoPE inside attention
        x_embed = self.input_embedding(x)

        # n transformer blocks
        for trans in self.transformer_blocks:
            x_embed = trans(x_embed)

        # Layer norm
        x_norm = self.final_ln(x_embed)

        # Output projection from d_model to vocab_size
        out = self.out_proj(x_norm)

        return out
