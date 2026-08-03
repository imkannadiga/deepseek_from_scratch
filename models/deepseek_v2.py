import torch
from models.blocks.rope_mla_transformer import RopeMLATransformer

class DeepSeek(torch.nn.Module):
    # MLA + DeepSeekMoE (fine-grained experts + shared expert isolation),
    # balanced with the standard auxiliary loss.
    def __init__(self, vocab_size, d_in, d_kv, max_seq_length, d_transformer, n_blocks, transformer_n_heads,
                 rope_head_dim=16,
                 n_experts_routed=16, top_k_routed=6, d_ff_routed=None,
                 n_experts_shared=1, d_ff_shared=None,
                 moe_loss="aux", gate="softmax", gamma=0.001, dropout=0.2):
        super().__init__()

        self.d_in = d_in
        self.d_kv = d_kv

        self.input_embedding = torch.nn.Embedding(vocab_size, d_in)

        self.n_blocks = n_blocks
        self.transformer_blocks = torch.nn.ModuleList([
            RopeMLATransformer(self.d_in, self.d_kv, d_transformer, transformer_n_heads,
                               rope_head_dim, max_seq_length, dropout,
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
