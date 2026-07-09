import torch
from models.layers.top_k_router import TopKRouter
from models.layers.expert import Expert

class MoE(torch.nn.Module):
    def __init__(self, embed_dim, n_experts, top_k):
        super().__init__()

        self.router = TopKRouter(embed_dim, n_experts, top_k)
        self.experts = torch.nn.ModuleList(*[Expert(embed_dim=embed_dim) for _ in range(n_experts)])

    def forward(self, x):
        # Get routing matrix from self.router

        routing_matrix, routing_indices = self.router(x)

        for i, expert in enumerate(self.experts):
            expert_mask = (routing_indices == i).any(dim=-1)