from torch import nn
from models.attention.ropeless_mla import RopelessMLA
from models.layers.sparse_moe import MoE

class RopelessMLATransformer(nn.Module):
    def __init__(self, d_in, d_kv, d_model, num_heads, dropout_p=0.2, num_experts=4, top_k=2):
        super().__init__()

        self.d_in = d_in
        self.d_kv = d_kv
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.ln_1 = nn.LayerNorm(d_in)

        self.attn = RopelessMLA(self.d_in, self.d_kv, self.d_model, self.d_model, self.num_heads)

        self.dropout = nn.Dropout(p=dropout_p)

        self.ln_2 = nn.LayerNorm(d_in)

        self.MoE = MoE(self.d_model, num_experts, top_k)

    def forward(self, x):
        B, n_tokens, d_in = x.shape

        # Layer Norm 1
        norm_x = self.ln_1(x)

        # Multi-head attention
        ctx = self.attn(norm_x)

        # Dropout
        ctx = self.dropout(ctx)

        # Residual
        x = x + ctx

        # Layer Norm 2
        norm_x = self.ln_2(x)

        # FFN 
        moe_out = self.MoE(norm_x)

        # Dropout
        moe_out = self.dropout(moe_out)

        # Residual
        x = x + moe_out

        return x