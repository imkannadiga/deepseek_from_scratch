import torch
from models.blocks.rope_mla_transformer import RopeMLATransformer
from models.layers.mtp import MultiTokenPredictionHead

class DeepSeek(torch.nn.Module):
    # Same MLA + DeepSeekMoE as V2, plus the two V3 additions: experts balanced by
    # a sigmoid-gated router whose per-expert bias is nudged toward even load (no
    # auxiliary loss term at all), and multi-token prediction.
    # depth=0 turns MTP off, which makes this V2 with aux-loss-free balancing.
    def __init__(self, vocab_size, d_in, d_kv, max_seq_length, d_transformer, n_blocks, transformer_n_heads,
                 rope_head_dim=16,
                 n_experts_routed=16, top_k_routed=6, d_ff_routed=None,
                 n_experts_shared=1, d_ff_shared=None,
                 moe_loss="free", gate="sigmoid", gamma=0.001, dropout=0.2, depth=0):
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

        # d_in, not d_transformer -- the head consumes trunk output and embeddings.
        # The MoE settings are passed through so the MTP blocks match the trunk.
        self.out_proj = MultiTokenPredictionHead(depth, d_in, vocab_size, transformer_n_heads,
                                                 d_kv, rope_head_dim, max_seq_length, dropout,
                                                 n_experts_routed, top_k_routed, d_ff_routed,
                                                 n_experts_shared, d_ff_shared,
                                                 moe_loss, gate, gamma)


    def forward(self, x, cache=None):
        B, T = x.shape

        # Token embedding only -- position is injected by RoPE inside attention
        x_embed = self.input_embedding(x)

        # n transformer blocks
        hidden = x_embed
        for layer_idx, trans in enumerate(self.transformer_blocks):
            hidden = trans(hidden, cache=cache, layer_idx=layer_idx)

        # Layer norm
        x_norm = self.final_ln(hidden)

        # Output projection from d_model to vocab_size
        out = self.out_proj(x_norm, x_embed)

        return out
