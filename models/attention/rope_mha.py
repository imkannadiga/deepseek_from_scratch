import torch
from models.embeddings.rope_embedding import RopeEmbedding

class RopeMHA(torch.nn.Module):
    def __init__(self, d_in, d_model, d_out, num_heads, max_seq_len=256):
        super().__init__()

        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.d_out = d_out
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim % 2 == 0, "head_dim should be even for RoPE"

        self.Q_weights = torch.nn.Linear(d_in, d_model, bias=False)
        self.K_weights = torch.nn.Linear(d_in, d_model, bias=False)
        self.V_weights = torch.nn.Linear(d_in, d_model, bias=False)

        self.rope = RopeEmbedding(self.head_dim, max_seq_len)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

        self.W_o = torch.nn.Linear(d_model, d_out, bias=False)

    def forward(self, x):
        B, num_tokens, d_in = x.shape

        # Pass through Q, K and V matrices
        Q = self.Q_weights(x)
        K = self.K_weights(x)
        V = self.V_weights(x)

        # Split into heads
        # (B, n_tokens, d_model) --> (B, n_heads, n_tokens, head_dim)
        Q = Q.view(B, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # RoPE replaces the additive positional embedding -- Q and K only
        Q = self.rope(Q)
        K = self.rope(K)

        # (B, n_heads, n_tokens, n_tokens)
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Mask upper triangle with -inf
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens] == 0.0, -torch.inf
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)

        ctx = attn_weights @ V

        # Stack heads back
        # (B, n_heads, n_tokens, head_dim) --> (B, n_tokens, d_model)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().reshape(B, num_tokens, self.d_model)

        return self.W_o(ctx)
