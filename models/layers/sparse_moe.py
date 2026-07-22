import torch
from models.layers.top_k_router import TopKRouter
from models.layers.expert import Expert

class MoE(torch.nn.Module):
    def __init__(self, embed_dim, n_experts, top_k, gamma=0.001):
        super().__init__()

        self.router = TopKRouter(embed_dim, n_experts, top_k)
        self.n_experts = n_experts
        self.gamma = gamma
        self.experts = torch.nn.ModuleList([*[Expert(embed_dim=embed_dim) for _ in range(n_experts)]])

    def forward(self, x):
        
        B, T, embed_dim = x.shape

        x_flat = x.view(B * T, embed_dim)

        routing_matrix, routing_indices = self.router(x_flat)

        final_output = torch.zeros_like(x_flat)

        expert_counts = torch.zeros(self.n_experts, device=x.device)

        for i, expert in enumerate(self.experts):
            expert_mask = (routing_indices == i).any(dim=-1)
            expert_counts[i] = (routing_indices == i).sum()
            if expert_mask.sum() == 0:
                continue

            expert_input = x_flat[expert_mask]                            # (n, embed_dim)
            expert_output = expert(expert_input)                          # (n, embed_dim)
            expert_weights = routing_matrix[expert_mask, i].unsqueeze(-1) # (n, 1)

            final_output[expert_mask] += expert_weights * expert_output

        self._update_bias(expert_counts)

        return final_output.view(B, T, embed_dim)

    @torch.no_grad()
    def _update_bias(self, expert_counts):
        total = expert_counts.sum()
        target = total / self.n_experts
        diff = expert_counts - target                   # positive = overloaded, negative = underloaded
        self.router.bias -= self.gamma * diff.sign()    # sign gives -1, 0, or +1
