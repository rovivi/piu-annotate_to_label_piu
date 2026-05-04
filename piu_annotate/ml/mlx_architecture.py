from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn


def sinusoidal_pos_encoding(seq_len: int, d_model: int) -> mx.array:
    positions = mx.arange(seq_len).reshape(-1, 1)
    div_term = mx.exp(mx.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = mx.concatenate([mx.sin(positions * div_term), mx.cos(positions * div_term)], axis=1)
    if d_model % 2 == 1:
        pe = mx.concatenate([pe, mx.zeros((seq_len, 1))], axis=1)
    return pe


class LimbSequenceTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 512,
        max_len: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = sinusoidal_pos_encoding(max_len, d_model)
        self.encoder = nn.TransformerEncoder(
            num_layers=n_layers,
            dims=d_model,
            num_heads=n_heads,
            mlp_dims=ffn_dim,
            dropout=dropout,
            checkpoint=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_head = nn.Linear(d_model, n_classes)
        self.d_model = d_model

    def __call__(
        self,
        x: mx.array,
        padding_mask: mx.array,
    ) -> mx.array:
        B, L, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_enc[:L][None, :, :]
        h = self.dropout(h)
        if mx.sum(padding_mask) > 0:
            attn_mask = mx.where(padding_mask[:, None, :, None], -1e9, 0.0)
        else:
            attn_mask = None
        h = self.encoder(h, mask=attn_mask)
        return self.out_head(h)