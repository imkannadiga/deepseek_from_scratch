import torch

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_in, d_model, d_out, num_heads):
    super().__init__()

    self.d_out = d_out
    self.d_model = d_model
    assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads

    self.Q_weights = torch.nn.Linear(d_in, d_model, bias=False)
    self.K_weights = torch.nn.Linear(d_in, d_model, bias=False)
    self.V_weights = torch.nn.Linear(d_in, d_model, bias=False)

    self.W_o = torch.nn.Linear(d_model, d_out, bias=False)

  def forward(self, x):
    B, num_tokens, d_in = x.shape
    # Pass through Q, K and V matrices
    Q = self.Q_weights(x)
    K = self.K_weights(x)
    V = self.V_weights(x)

    # Split Q, K and V into num heads
    # (B, num_tokens, d_out) -> (B, num_tokens, num_heads, head_dim)
    Q = Q.view(B, num_tokens, self.num_heads, self.head_dim)
    K = K.view(B, num_tokens, self.num_heads, self.head_dim)
    V = V.view(B, num_tokens, self.num_heads, self.head_dim)

    # Group by the numver if heads
    # Currently the data is like
    # (Q1H1, Q2H1, Q3H1....)
    # We want
    # (Q1H1, Q2H1, ...Q1H2, Q2H2...)
    Q = Q.permute(0, 2, 1, 3)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)

    # Now the shape is (B, num_heads, num_tokens, head_dim)

    # Compute Q*K_trans on the num_heads
    attn_scores = Q @ K.transpose(-2, -1)

    attn_scores = attn_scores / K.shape[-1] ** 0.5

    # The shape is (B, num_heads, num_tokens, num_tokens)

    # Mask upper triangle and replace with -inf

    mask = torch.ones((num_tokens, num_tokens))
    mask = torch.tril(mask)

    masked_attn_scores = attn_scores.masked_fill(mask==0.0, -torch.inf)

    A = torch.softmax(masked_attn_scores, dim=-1)
    ctx_unstacked = A @ V

    # Now, the unstacked context vector is of the shape
                                    # (B, num_heads, num_tokens, head_dim)
    # But the attention score should be of the format
                                    # (B, num_tokens, d_out)
                                    # = (B, num_tokens, num_heads*head_dim)

    # 1. Reshape to get num_tokens at 2nd dim
    ctx_unstacked = ctx_unstacked.permute(0,2,1,3).contiguous()

    # 2. Stack on the num_heads dim
    ctx_stacked = ctx_unstacked.reshape(B, num_tokens, self.d_out)

    ctx_out = self.W_o(ctx_stacked)

    return ctx_out