import torch
from models.embeddings.rope_embedding import RopeEmbedding

class RopeMLA(torch.nn.Module):
    # Decoupled RoPE: every head is split into a NoPE part that comes out of the
    # compressed KV latent, and a small RoPE part computed straight from x.
    # The RoPE key is a single head shared by all heads, so the latent stays cacheable.
    #
    # At decode time the up-projections are absorbed into the neighbouring
    # matrices (see _absorbed_matrices), so K and V are never materialised.
    def __init__(self, d_in, d_kv, d_model, d_out, n_heads, rope_head_dim=16, max_seq_len=256):
        super().__init__()

        assert d_model % n_heads == 0, "d_model should be a multiple of n_heads"
        assert rope_head_dim % 2 == 0, "rope_head_dim should be even for RoPE"

        self.d_model = d_model
        self.d_kv = d_kv
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

        # Lazily built products of the frozen weights, see _absorbed_matrices
        self._absorbed = None

    def train(self, mode=True):
        # The weights move during training, so any absorbed product is stale
        self._absorbed = None
        return super().train(mode)

    def _absorbed_matrices(self):
        """
        The absorption trick. Scores and outputs both factor so that the
        up-projections can be folded into their neighbours:

            (x W_q)(c W_uk)^T = x (W_q W_uk^T) c^T      ->  A, per head
            (alpha c W_uv) W_o = (alpha c) (W_uv W_o)   ->  B, per head

        A pulls queries straight into the compressed space and B takes the
        compressed context straight to the output, so K and V never exist.
        Built once and reused across decode steps; cleared by .train().

        Both are stored with the head axis folded into a single matrix so each
        one is one GEMM rather than n_heads small ones -- that difference is
        worth more than the FLOPs at this size.
        """
        if self._absorbed is None:
            # Linear stores weight as (out_features, in_features)
            Wq = self.W_q.weight.view(self.n_heads, self.head_dim, -1)    # (h, head_dim, d_in)
            Wuk = self.W_uk.weight.view(self.n_heads, self.head_dim, -1)  # (h, head_dim, d_kv)
            A = Wq.transpose(1, 2) @ Wuk                                  # (h, d_in, d_kv)
            A = A.permute(1, 0, 2).reshape(-1, self.n_heads * self.d_kv)  # (d_in, h*d_kv)

            Wuv = self.W_uv.weight.view(self.n_heads, self.head_dim, -1)  # (h, head_dim, d_kv)
            Wo = self.W_o.weight.view(-1, self.n_heads, self.head_dim)    # (d_out, h, head_dim)
            B = Wuv.transpose(1, 2) @ Wo.permute(1, 2, 0)                 # (h, d_kv, d_out)
            B = B.reshape(self.n_heads * self.d_kv, -1)                   # (h*d_kv, d_out)

            self._absorbed = (A, B)

        return self._absorbed

    def forward(self, x, cache=None, layer_idx=None):
        B, num_tokens, d_in = x.shape

        # How many tokens are already cached -- 0 during training and prefill
        past_len = cache.get_seq_length(layer_idx) if cache is not None else 0
        total_len = past_len + num_tokens
        assert total_len <= self.mask.shape[0], \
            f"context {total_len} exceeds max_seq_len {self.mask.shape[0]}"

        # Step : 1
        # Compress x into the KV latent
        # (B, n_tokens, d_in) --> (B, n_tokens, d_kv)
        c_kv = self.W_dkv(x)

        # Step : 2
        # Shared RoPE key head, rotated at its absolute position before caching
        # (B, n_tokens, d_in) --> (B, 1, n_tokens, rope_head_dim)
        K_rope = self.W_kr(x).view(B, num_tokens, 1, self.rope_head_dim).permute(0, 2, 1, 3)
        K_rope = self.rope(K_rope, offset=past_len)

        # Step : 3
        # K_rope is built from x rather than from the latent, so it cannot be
        # recovered once x is gone -- it rides along in the same cache entry.
        # d_kv + rope_head_dim per token per layer is exactly what MLA caches.
        entry = torch.cat([
            c_kv,
            K_rope.permute(0, 2, 1, 3).reshape(B, num_tokens, self.rope_head_dim),
        ], dim=-1)

        if cache is not None:
            entry = cache.update(layer_idx, entry)   # (B, total_len, d_kv + rope_head_dim)

        c_kv = entry[..., :-self.rope_head_dim]
        K_rope = entry[..., -self.rope_head_dim:] \
            .view(B, total_len, 1, self.rope_head_dim).permute(0, 2, 1, 3)

        # Step : 4
        # Queries carry the position, and are the same in both paths
        Q_rope = self.W_qr(x).view(B, num_tokens, self.n_heads, self.rope_head_dim).permute(0, 2, 1, 3)
        Q_rope = self.rope(Q_rope, offset=past_len)

        # Step : 5
        # Absorption only pays off when there is a single query row. For a long
        # prefill the score matmul would run over d_kv instead of head_dim, which
        # is wider -- so prefill and training take the plain path.
        if not self.training and num_tokens == 1:
            return self._attend_absorbed(x, c_kv, K_rope, Q_rope, past_len, total_len)

        return self._attend_plain(x, c_kv, K_rope, Q_rope, past_len, total_len)

    def _attend_plain(self, x, c_kv, K_rope, Q_rope, past_len, total_len):
        B, num_tokens, _ = x.shape

        # NoPE half of K and V from the full latent history, Q from x only
        # (B, total_len, d_model) --> (B, n_heads, total_len, head_dim)
        K_nope = self.W_uk(c_kv).view(B, total_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.W_uv(c_kv).view(B, total_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Q_nope = self.W_q(x).view(B, num_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Broadcast the shared key head over all heads, then glue the halves
        # (B, n_heads, *, head_dim + rope_head_dim)
        Q = torch.cat([Q_nope, Q_rope], dim=-1)
        K = torch.cat([K_nope, K_rope.expand(-1, self.n_heads, -1, -1)], dim=-1)

        # (B, n_heads, n_tokens, total_len)
        attn_scores = (Q @ K.transpose(-2, -1)) / ((self.head_dim + self.rope_head_dim) ** 0.5)
        attn_weights = torch.softmax(self._mask(attn_scores, past_len, total_len), dim=-1)

        ctx_vector = attn_weights @ V

        # (B, n_heads, n_tokens, head_dim) --> (B, n_tokens, d_model) --> (B, n_tokens, d_out)
        ctx_vector = ctx_vector.permute(0, 2, 1, 3).contiguous().reshape(B, num_tokens, self.d_model)
        return self.W_o(ctx_vector)

    def _attend_absorbed(self, x, c_kv, K_rope, Q_rope, past_len, total_len):
        B, num_tokens, _ = x.shape
        A, B_mat = self._absorbed_matrices()

        # Queries go straight into the compressed space -- W_uk never runs, so
        # K is never built. (B, n_tokens, d_in) --> (B, n_heads, n_tokens, d_kv)
        Q_nope = (x @ A).view(B, num_tokens, self.n_heads, self.d_kv).permute(0, 2, 1, 3)

        # Both terms are exactly the two halves of the plain path's dot product,
        # so they share its single denominator. Scaling them apart would stop
        # this being the same function.
        c_kv = c_kv.unsqueeze(1)                                  # (B, 1, total_len, d_kv)
        attn_scores = (Q_nope @ c_kv.transpose(-2, -1)
                       + Q_rope @ K_rope.transpose(-2, -1))
        attn_scores = attn_scores / ((self.head_dim + self.rope_head_dim) ** 0.5)

        attn_weights = torch.softmax(self._mask(attn_scores, past_len, total_len), dim=-1)

        # Context stays compressed and B takes it to the output in one step --
        # W_uv and W_o never run. (B, n_heads, n_tokens, d_kv) --> (B, n_tokens, d_out)
        ctx_vector = attn_weights @ c_kv
        ctx_vector = ctx_vector.permute(0, 2, 1, 3).reshape(B, num_tokens, self.n_heads * self.d_kv)
        return ctx_vector @ B_mat

    def _mask(self, attn_scores, past_len, total_len):
        # Queries sit at absolute positions past_len..total_len-1, keys at 0..total_len-1
        return attn_scores.masked_fill(
            self.mask[past_len:total_len, :total_len] == 0.0, -torch.inf
        )
