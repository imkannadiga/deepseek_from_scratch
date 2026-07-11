import torch
from models.layers.top_k_router import TopKRouter
from models.layers.expert import Expert

class MoE(torch.nn.Module):
    def __init__(self, embed_dim, n_experts, top_k):
        super().__init__()

        self.router = TopKRouter(embed_dim, n_experts, top_k)
        self.experts = torch.nn.ModuleList([*[Expert(embed_dim=embed_dim) for _ in range(n_experts)]])

    def forward(self, x):
        
        B, T, embed_dim = x.shape

        x_flat = x.view(B * T, embed_dim)

        routing_matrix, routing_indices = self.router(x_flat)

        final_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            expert_mask = (routing_indices == i).any(dim=-1)

            if expert_mask.sum() == 0:
                continue

            expert_input = x_flat[expert_mask]                            # (n, embed_dim)
            expert_output = expert(expert_input)                          # (n, embed_dim)
            expert_weights = routing_matrix[expert_mask, i].unsqueeze(-1) # (n, 1)

            final_output[expert_mask] += expert_weights * expert_output

        return final_output.view(B, T, embed_dim)