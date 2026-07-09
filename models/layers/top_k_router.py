import torch

class TopKRouter(torch.nn.Module):
    def __init__(self, n_embed, n_experts, top_k):
        super().__init__()

        self.top_k = top_k
        self.router = torch.nn.Linear(n_embed, n_experts)

    def forward(self, x):

        routing_matrix = self.router(x)

        top_k_logits, top_k_posn = routing_matrix.topk(self.top_k, dim=-1)
        zeros = torch.full_like(routing_matrix, float('-inf'))
        sparse_logits = zeros.scatter(-1, top_k_posn, top_k_logits)

        return torch.nn.functional.softmax(sparse_logits, dim=-1), top_k_posn