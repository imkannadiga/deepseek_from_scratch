import torch
from models.embeddings.rope_embedding import RopeEmbedding

class RopeMLA(torch.nn.Module):
    # Decoupled RoPE: every head is split into a NoPE part that comes out of the
    # compressed KV latent, and a small RoPE part computed straight from x.
    # The RoPE key is a single head shared by all heads, so the latent stays cacheable.
    def __init__(self, d_in, d_kv, d_model, d_out, n_heads, rope_head_dim=16, max_seq_len=256):
        super().__init__()

        assert d_model % n_heads == 0, "d_model should be a multiple of n_heads"
        assert rope_head_dim % 2 == 0, "rope_head_dim should be even for RoPE"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_head_dim = rope_head_dim

        # Compressed KV latent -> NoPE keys and values
        self.W_dkv = torch.nn.Linear(d_in, d_kv, bias=False)
        self.W_uk = torch.nn.Linear(d_kv, d_model, bias=False)
        self.W_uv = torch.nn.Linear(d_kv, d_model, bias=False)

        # NoPE queries
        self.W_q = torch.nn.Linear(d_in, d_model, bias=False)

        # RoPE halves -- per head for Q, one shared head for K
        self.W_qr = torch.nn.Linear(d_in, n_heads * rope_head_dim, bias=False)
        self.W_kr = torch.nn.Linear(d_in, rope_head_dim, bias=False)

        self.rope = RopeEmbedding(rope_head_dim, max_seq_len)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

        self.W_o = torch.nn.Linear(d_model, d_out, bias=False)

    def forward(self, x):
        B, num_tokens, d_in = x.shape

        # Step : 1
        # Compute KV cache by passing x through W_dkv
        # (B, n_tokens, d_in) --> (B, n_tokens, d_kv)
        c_kv = self.W_dkv(x)

        ####
        # HERE IS WHERE ACTUAL CACHING LOGIC NEEDS TO BE IMPLEMENTED
        # DURING INFERENCE, X WILL JUST BE THE LAST TOKEN
        # STEPS - GET C_KV, APPEND IT TO PRE_CACHED_DATA, CONTINUE WITH REST
        ####

        # Step : 2
        # NoPE half of K, V and Q, split into heads
        # (B, n_tokens, d_model) --> (B, n_heads, n_tokens, head_dim)
        K_nope = self.W_uk(c_kv).view(B, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_uv(c_kv).view(B, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Q_nope = self.W_q(x).view(B, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Step : 3
        # RoPE half -- Q gets one per head, K gets a single head
        # (B, n_tokens, d_in) --> (B, n_heads or 1, n_tokens, rope_head_dim)
        Q_rope = self.W_qr(x).view(B, num_tokens, self.n_heads, self.rope_head_dim).permute(0, 2, 1, 3)
        K_rope = self.W_kr(x).view(B, num_tokens, 1, self.rope_head_dim).permute(0, 2, 1, 3)

        # Step : 4
        # Rotate the RoPE half, then broadcast the shared key head over all heads
        Q_rope = self.rope(Q_rope)
        K_rope = self.rope(K_rope).expand(-1, self.n_heads, -1, -1)

        # Step : 5
        # Glue the halves back together
        # (B, n_heads, n_tokens, head_dim + rope_head_dim)
        Q = torch.cat([Q_nope, Q_rope], dim=-1)
        K = torch.cat([K_nope, K_rope], dim=-1)

        # Step : 6
        # Scaled dot product over the combined head dim
        # (B, n_heads, n_tokens, n_tokens)
        attn_scores = (Q @ K.transpose(-2, -1)) / ((self.head_dim + self.rope_head_dim) ** 0.5)

        # Step : 7
        # Mask upper triangle of attn_scores with -inf
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens] == 0.0, -torch.inf
        )

        # Step : 8
        # Softmax then weight the values
        # (B, n_heads, n_tokens, n_tokens) * (B, n_heads, n_tokens, head_dim) --> (B, n_heads, n_tokens, head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        ctx_vector = attn_weights @ V

        # Step : 9
        # Stack over n_heads
        # (B, n_heads, n_tokens, head_dim) --> (B, n_tokens, d_model)
        ctx_vector = ctx_vector.permute(0, 2, 1, 3).contiguous().reshape(B, num_tokens, self.d_model)

        # Step : 10
        # Pass context vector through W_o to get final output projections
        # (B, n_tokens, d_model) --> (B, n_tokens, d_out)
        return self.W_o(ctx_vector)
