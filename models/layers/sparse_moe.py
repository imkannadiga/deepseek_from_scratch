import torch
from models.layers.top_k_router import TopKRouter
from models.layers.expert import Expert

class MoE(torch.nn.Module):
    def __init__(self, n_experts_routed, d_ff_routed, top_k_routed, n_experts_shared, embed_dim, d_ff_shared,
                 dropout=0.1, gamma=0.001, moe_loss="aux", gate="softmax"):
        super().__init__()

        self.router = TopKRouter(embed_dim, n_experts_routed, top_k_routed, gate)
        self.n_experts_routed = n_experts_routed
        self.n_experts_shared = n_experts_shared
        self.top_k_routed = top_k_routed

        self.gamma = gamma
        self.moe_loss = moe_loss
        self.aux_loss = None

        self.experts_routed = torch.nn.ModuleList([Expert(embed_dim, d_ff_routed, dropout) for _ in range(n_experts_routed)])
        self.experts_shared = torch.nn.ModuleList([Expert(embed_dim, d_ff_shared, dropout) for _ in range(n_experts_shared)])

    def forward(self, x):

        B, T, embed_dim = x.shape

        x_flat = x.view(B * T, embed_dim)

        routing_matrix, routing_indices, affinity = self.router(x_flat)

        final_output = torch.zeros_like(x_flat)

        expert_counts = torch.zeros(self.n_experts_routed, device=x.device)

        # Routed experts
        for i, expert in enumerate(self.experts_routed):
            expert_mask = (routing_indices == i).any(dim=-1)
            expert_counts[i] = (routing_indices == i).sum()
            if expert_mask.sum() == 0:
                continue

            expert_input = x_flat[expert_mask]                            # (n, embed_dim)
            expert_output = expert(expert_input)                          # (n, embed_dim)
            expert_weights = routing_matrix[expert_mask, i].unsqueeze(-1) # (n, 1)

            final_output[expert_mask] += expert_weights * expert_output

        # Shared experts -- always on, no gate, every token pays for them
        for expert in self.experts_shared:
            final_output += expert(x_flat)

        # Balancing is a training-time concern only. V2 pays for it with a loss
        # term; V3 gets it free by nudging the router bias instead.
        self.aux_loss = None
        if self.training:
            if self.moe_loss == "free":
                self._update_bias(expert_counts)
            else:
                self.aux_loss = self._expert_balance_loss(affinity, expert_counts)

        return final_output.view(B, T, embed_dim)

    @torch.no_grad()
    def _update_bias(self, expert_counts):
        total = expert_counts.sum()
        target = total / self.n_experts_routed
        diff = expert_counts - target                   # positive = overloaded, negative = underloaded
        self.router.bias -= self.gamma * diff.sign()    # sign gives -1, 0, or +1

    def _expert_balance_loss(self, affinity, expert_counts):
        # DeepSeek-V2 expert-level balance loss. f counts how often each expert was
        # picked (no gradient), P is its mean affinity (carries the gradient).
        # Both terms are flat when load is even, so this bottoms out near 1.0.
        n_tokens = affinity.shape[0]
        f = expert_counts * self.n_experts_routed / (self.top_k_routed * n_tokens)
        P = affinity.mean(dim=0)
        return (f * P).sum()


def collect_aux_loss(model):
    # Sum the balance loss across every MoE layer. None when nothing produced one,
    # which is the case in eval and for any aux-loss-free model.
    losses = [m.aux_loss for m in model.modules() if isinstance(m, MoE) and m.aux_loss is not None]
    return sum(losses) if losses else None
