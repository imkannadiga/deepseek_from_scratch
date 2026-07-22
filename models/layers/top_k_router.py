import torch

class TopKRouter(torch.nn.Module):
    def __init__(self, n_embed, n_experts, top_k, gamma=0.0001):
        super().__init__()

        self.top_k = top_k
        self.router = torch.nn.Linear(n_embed, n_experts)

        self.register_buffer("bias", torch.zeros(n_experts))
        
    def forward(self, x):

        routing_matrix = self.router(x)
        biased_logits = routing_matrix + self.bias

        _, top_k_posn = biased_logits.topk(self.top_k, dim=-1) 
        zeros = torch.full_like(routing_matrix, float('-inf'))
        sparse_logits = zeros.scatter( 
            -1, top_k_posn,
            routing_matrix.gather(-1, top_k_posn)
        )

        return torch.nn.functional.softmax(sparse_logits, dim=-1), top_k_posn