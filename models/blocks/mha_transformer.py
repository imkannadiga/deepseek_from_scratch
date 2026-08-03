from torch import nn
from models.attention.multi_head_attention import MultiHeadAttention

class MHATransformer(nn.Module):
    def __init__(self, d_in, d_model, num_heads, d_ff=None, dropout_p=0.2):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        d_ff = d_ff if d_ff is not None else 4 * d_in

        self.ln_1 = nn.LayerNorm(d_in)

        self.attn = MultiHeadAttention(self.d_in, self.d_model, self.d_in, self.num_heads)

        self.dropout = nn.Dropout(p=dropout_p)

        self.ln_2 = nn.LayerNorm(d_in)

        self.ffn = nn.Sequential(
            nn.Linear(d_in, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_in),
        )

    def forward(self, x, cache=None, layer_idx=None):
        B, n_tokens, d_in = x.shape

        # Layer Norm 1
        norm_x = self.ln_1(x)

        # Multi-head attention
        ctx = self.attn(norm_x, cache, layer_idx)

        # Dropout
        ctx = self.dropout(ctx)

        # Residual
        x = x + ctx

        # Layer Norm 2
        norm_x = self.ln_2(x)

        # FFN 
        ffn_out = self.ffn(norm_x)

        # Dropout
        ffn_out = self.dropout(ffn_out)

        # Residual
        x = x + ffn_out

        return x