import torch

class SelfAttention(torch.nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()

    self.d_out = d_out

    self.Q_weights = torch.nn.Linear(d_in, d_out, bias=False)
    self.K_weights = torch.nn.Linear(d_in, d_out, bias=False)
    self.V_weights = torch.nn.Linear(d_in, d_out, bias=False)

  def forward(self, x):
    B, num_tokens, d_in = x.shape
    # Pass through Q, K and V matrices
    Q = self.Q_weights(x)
    K = self.K_weights(x)
    V = self.V_weights(x)
    # Compute Q*K_trans
    A = Q @ K.transpose(-2, -1)
    A = torch.softmax(A / K.shape[-1] ** 0.5, dim=-1)
    return A @ V