from torch import nn
from models.attention.rope_mha import RopeMHA
from models.layers.sparse_moe import MoE

class RopeMHATransformer(nn.Module):
    def __init__(self, d_in, d_model, num_heads, max_seq_len=256, dropout_p=0.2,
                 n_experts_routed=8, top_k_routed=2, d_ff_routed=None,
                 n_experts_shared=0, d_ff_shared=None,
                 moe_loss="aux", gate="softmax", gamma=0.001):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.ln_1 = nn.LayerNorm(d_in)

        self.attn = RopeMHA(self.d_in, self.d_model, self.d_in, self.num_heads, max_seq_len)

        self.dropout = nn.Dropout(p=dropout_p)

        self.ln_2 = nn.LayerNorm(d_in)

        self.MoE = MoE(n_experts_routed, d_ff_routed, top_k_routed,
                       n_experts_shared, self.d_in, d_ff_shared,
                       dropout_p, gamma, moe_loss, gate)

    def forward(self, x, cache=None, layer_idx=None):
        B, n_tokens, d_in = x.shape

        # Layer Norm 1
        norm_x = self.ln_1(x)

        # Multi-head attention with RoPE
        ctx = self.attn(norm_x, cache, layer_idx)

        # Dropout
        ctx = self.dropout(ctx)

        # Residual
        x = x + ctx

        # Layer Norm 2
        norm_x = self.ln_2(x)

        # MoE instead of dense FFN
        moe_out = self.MoE(norm_x)

        # Dropout
        moe_out = self.dropout(moe_out)

        # Residual
        x = x + moe_out

        return x
