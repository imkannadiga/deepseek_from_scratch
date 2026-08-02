import torch

class TopKRouter(torch.nn.Module):
    def __init__(self, n_embed, n_experts, top_k, gate="softmax"):
        super().__init__()

        self.top_k = top_k
        self.gate = gate
        self.router = torch.nn.Linear(n_embed, n_experts)

        # Only aux-loss-free balancing moves this; it stays zero otherwise
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x):
        logits = self.router(x)

        # V2 and earlier score experts with a softmax, V3 switched to sigmoid
        if self.gate == "sigmoid":
            affinity = torch.sigmoid(logits)
        else:
            affinity = torch.softmax(logits, dim=-1)

        # Bias steers selection only -- the gate values stay unbiased
        _, top_k_posn = (affinity + self.bias).topk(self.top_k, dim=-1)
        top_k_scores = affinity.gather(-1, top_k_posn)

        # Sigmoid gates do not sum to 1 on their own, so V3 renormalises them
        if self.gate == "sigmoid":
            top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-9)

        routing_matrix = torch.zeros_like(affinity).scatter(-1, top_k_posn, top_k_scores)

        # affinity is the dense score over every expert -- the balance loss needs it
        return routing_matrix, top_k_posn, affinity
