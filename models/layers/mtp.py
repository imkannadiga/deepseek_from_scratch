import torch
from models.blocks.rope_mla_transformer import RopeMLATransformer

class MultiTokenPredictionHead(torch.nn.Module):
    # Sequential MTP: depth k takes the previous depth's hidden state, fuses it
    # with the embedding of the token k steps ahead, and predicts one token
    # further out. The un-embedding is shared across every depth.
    #
    # The extra depths only run while training. At eval the head is just the
    # output projection, so generation and sampling see a plain (B, T, V) tensor
    # and pay nothing for MTP.
    def __init__(self, depth, d_model, vocab_size, n_heads=4, d_kv=None, rope_head_dim=16, max_seq_len=256,
                 dropout=0.2,
                 n_experts_routed=16, top_k_routed=6, d_ff_routed=None,
                 n_experts_shared=1, d_ff_shared=None,
                 moe_loss="free", gate="sigmoid", gamma=0.001):
        super().__init__()

        self.depth = depth
        self.mtp_logits = None

        self.out_proj = torch.nn.Linear(d_model, vocab_size, bias=False)

        if depth > 0:
            d_kv = d_kv if d_kv is not None else d_model // 2

            # one norm pair + projection + transformer block per depth level
            self.norms_emb  = torch.nn.ModuleList([
                torch.nn.RMSNorm(d_model) for _ in range(depth)
            ])
            self.norms_hidden  = torch.nn.ModuleList([
                torch.nn.RMSNorm(d_model) for _ in range(depth)
            ])

            self.projections = torch.nn.ModuleList([
                torch.nn.Linear(2*d_model, d_model, bias=False)
                for _ in range(depth)
            ])

            # Takes the fused vector, which the projection has already brought
            # back down to d_model. The block carries its own MoE, so the MoE
            # settings have to mirror the trunk -- otherwise these blocks fall
            # back to aux-loss balancing and quietly put a balance loss back
            # into a model whose whole point is not having one.
            self.blocks = torch.nn.ModuleList([
                RopeMLATransformer(d_in = d_model,
                        d_kv = d_kv,
                        d_model = d_model,
                        num_heads = n_heads,
                        rope_head_dim = rope_head_dim,
                        max_seq_len = max_seq_len,
                        dropout_p = dropout,
                        n_experts_routed = n_experts_routed,
                        top_k_routed = top_k_routed,
                        d_ff_routed = d_ff_routed,
                        n_experts_shared = n_experts_shared,
                        d_ff_shared = d_ff_shared,
                        moe_loss = moe_loss,
                        gate = gate,
                        gamma = gamma) for _ in range(depth)
            ])


    def forward(self, x, embeddings):
        # Main next-token prediction -- always produced
        main_logits = self.out_proj(x)

        if self.depth == 0 or not self.training:
            self.mtp_logits = None
            return main_logits

        mtp_logits = []

        # Initial hidden state z_0
        h = x
        for k in range(self.depth):
            # Truncate to T-k-1 since we can only
            # predict current depth for those tokens
            h_trunc   = h[:, :-1, :]
            emb_shift = embeddings[:, k+1:, :]

            # RMS norm [RMSNorm(h_k-1) ; RMSNorm(inp_emb_k)]
            norm_hidden = self.norms_hidden[k](h_trunc)
            norm_emb = self.norms_emb[k](emb_shift)
            fused = torch.cat([norm_hidden, norm_emb], dim=-1)

            # Project fused norms back to d_model
            fused = self.projections[k](fused)

            # Transformer block
            h = self.blocks[k](fused)

            # Output projection from common un-embedding layer
            mtp_logits.append(self.out_proj(h))

        # Stashed rather than returned so every model in the repo keeps the same
        # forward contract -- the trainer picks these up via collect_mtp_logits
        self.mtp_logits = mtp_logits

        return main_logits


def collect_mtp_logits(model):
    # Per-depth logits from the MTP head, or None when the model has no head or
    # is in eval. Depth d covers T-(d+1) positions and predicts d+1 tokens ahead.
    for m in model.modules():
        if isinstance(m, MultiTokenPredictionHead):
            return m.mtp_logits
    return None
