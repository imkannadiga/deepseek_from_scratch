import torch
from models.blocks.rope_mha_transformer import RopeMHATransformer

class DeepSeek(torch.nn.Module):
    # MHA + RoPE + a plain top-k MoE: one expert size, no shared experts.
    def __init__(self, vocab_size, d_in, max_seq_length, d_transformer, n_blocks, transformer_n_heads,
                 n_experts_routed=8, top_k_routed=2, d_ff_routed=None,
                 n_experts_shared=0, d_ff_shared=None,
                 moe_loss="aux", gate="softmax", gamma=0.001, dropout=0.2):
        super().__init__()

        self.d_in = d_in

        self.input_embedding = torch.nn.Embedding(vocab_size, d_in)

        self.n_blocks = n_blocks
        self.transformer_blocks = torch.nn.ModuleList([
            RopeMHATransformer(self.d_in, d_transformer, transformer_n_heads, max_seq_length, dropout,
                               n_experts_routed, top_k_routed, d_ff_routed,
                               n_experts_shared, d_ff_shared,
                               moe_loss, gate, gamma)
            for _ in range(n_blocks)
        ])

        self.final_ln = torch.nn.LayerNorm(d_in)

        self.out_proj = torch.nn.Linear(d_in, vocab_size)


    def forward(self, x, cache=None):
        B, T = x.shape

        # Token embedding only -- position is injected by RoPE inside attention
        x_embed = self.input_embedding(x)

        # n transformer blocks
        for layer_idx, trans in enumerate(self.transformer_blocks):
            x_embed = trans(x_embed, cache=cache, layer_idx=layer_idx)

        # Layer norm
        x_norm = self.final_ln(x_embed)

        # Output projection from d_model to vocab_size
        out = self.out_proj(x_norm)

        return out
