import torch

class RopelessMLA(torch.nn.Module):
    def __init__(self, d_in, d_kv, d_model, d_out, n_heads):
        super().__init__()

        self.W_dkv = torch.nn.Linear(d_in, d_kv, bias=False)

        self.W_uk = torch.nn.Linear(d_kv, d_model, bias=False)
        self.W_uv = torch.nn.Linear(d_kv, d_model, bias=False)

        self.W_q = torch.nn.Linear(d_in, d_model, bias=False)

        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model should be a multiple of n_heads"

        self.head_dim = self.d_model // self.n_heads


        self.W_o = torch.nn.Linear(d_model, d_out, bias=False)


    def forward(self, x):
        B, num_tokens, d_in = x.shape
        device = x.device

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
        # Compute K and V by passing c_kv thorugh W_uk and W_Uv
        # (B, n_tokens, d_kv) --> (B, n_tokens, d_model)
        K = self.W_uk(c_kv)
        V = self.W_uv(c_kv)
        
        # Step : 3
        # Compute Q by passing x through W_q
        # (B, n_tokens, d_in) --> (B, n_tokens, d_model)
        Q = self.W_q(x)
        
        # Step : 4 
        # Separate Q, K and V into n_heads 
        # (B, n_tokens, d_model) --> (B, n_tokens, n_heads, head_dim)
        Q = Q.view(B, num_tokens, self.n_heads, self.head_dim)
        K = K.view(B, num_tokens, self.n_heads, self.head_dim)
        V = V.view(B, num_tokens, self.n_heads, self.head_dim)           

        # Step : 5
        # View Q, K and V by multiple heads
        # (B, n_tokens, n_heads, head_dim) --> (B, n_heads, n_tokens, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Step : 6
        # Multiply Q with K_t to get raw_attn_scores
        # (B, n_heads, n_tokens, head_dim) * (B, n_heads, head_dim, n_tokens) --> (B, n_heads, n_tokens, n_tokens)
        raw_attn_scores = Q @ K.transpose(-2, -1)

        # Step : 7
        # Divide raw_attn_scores by sqrt(head_dim) to get attn_scores
        attn_scores = raw_attn_scores / (self.head_dim ** 0.5)

        # Step : 8
        # Mask upper triangle of attn_scores with -inf
        mask = torch.tril(torch.ones((num_tokens, num_tokens), device=device))
        attn_scores = attn_scores.masked_fill(mask==0.0, -torch.inf)

        # Step : 9
        # Pass attn_scores through Softmax to get values between (0, 1) to get attn_weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Step : 10
        # Multiply attn_weights with V to get context_vector
        # (B, n_heads, n_tokens, n_tokens) * (B, n_heads, n_tokens, head_dim) --> (B, n_heads, n_tokens, head_dim)
        ctx_vector = attn_weights @ V

        # Step : 11
        # Reshape to group back by n_tokens
        # (B, n_heads, n_tokens, head_dim) --> (B, n_tokens, n_heads, head_dim)
        ctx_vector = ctx_vector.permute(0, 2, 1 ,3).contiguous()

        # Step : 12
        # Stack over n_heads to get final context vector
        # (B, n_tokens, n_heads, head_dim) --> (B, n_tokens, n_heads*head_dim=d_model)
        ctx_vector = ctx_vector.reshape(B, num_tokens, self.d_model)

        # Step : 13
        # Pass context vector through W_o to get final output projections
        # (B, n_tokens, d_model) --> (B, n_tokens, d_out)
        final_proj = self.W_o(ctx_vector)

        return final_proj