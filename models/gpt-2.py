import torch
from models.blocks.mha_transformer import MHATransformer
from models.embeddings.sin_embedding import SinosoidalPositionalEmbedding

class GPT2(torch.nn.Module):
    def __init__(self, vocab_size, d_in, max_seq_length, d_transformer, n_blocks, transformer_n_heads):
        super().__init__()

        self.d_in = d_in

        self.input_embedding = torch.nn.Embedding(vocab_size, d_in)  
        self.pos_embedding = SinosoidalPositionalEmbedding(max_seq_length, d_in)

        self.n_blocks = n_blocks
        self.transformer_blocks = torch.nn.ModuleList([
            MHATransformer(self.d_in, d_transformer, transformer_n_heads)
            for _ in range(n_blocks)
        ])

        self.final_ln = torch.nn.LayerNorm(d_in)

        self.out_proj = torch.nn.Linear(d_in, vocab_size)

        
    def forward(self, x):
        B, T = x.shape

        # Token embedding + Positional embedding
        x_embed = self.input_embedding(x) + self.pos_embedding(x)

        # N transformer blocks
        for trans in self.transformer_blocks:
            x_embed = trans(x_embed)

        # Layer norm
        x_norm = self.final_ln(x_embed)

        # Output projection from d_model to vocab_size
        out = self.out_proj(x_norm)

        return out