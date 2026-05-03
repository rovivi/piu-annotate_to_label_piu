from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

class LimbTransformer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        model_dim: int = 64, 
        num_heads: int = 4, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Embedding(50, model_dim) # Max context length 50
        
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=model_dim,
            num_heads=num_heads,
            mlp_dims=model_dim * 4,
            checkpoint=False,
        )
        
        self.out_proj = nn.Linear(model_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        
        seq_len = x.shape[1]
        pos = mx.arange(seq_len)
        x = x + self.pos_embedding(pos)
        
        x = self.dropout(x)
        x = self.transformer(x, mask=mask)
        
        # We only care about the prediction for the target note (usually the center of the sequence)
        # But for training efficiency, we might predict for all.
        # Let's assume we take the last hidden state for now if it's a sliding window, 
        # or all if it's a sequence model.
        return self.out_proj(x)
