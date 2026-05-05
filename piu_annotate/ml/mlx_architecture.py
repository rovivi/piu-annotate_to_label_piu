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


class OutputHead(nn.Module):
    """Two-layer classifier with GELU activation."""
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, n_classes)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class LimbSequenceTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        max_len: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        # Normalizes mixed feature scales (timing floats vs boolean 0/1)
        self.input_norm = nn.LayerNorm(input_dim)
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
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = OutputHead(d_model, n_classes, dropout)
        self.d_model = d_model

    def __call__(
        self,
        x: mx.array,
        padding_mask: mx.array,
    ) -> mx.array:
        B, L, _ = x.shape
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = h + self.pos_enc[:L][None, :, :]
        h = self.dropout(h)
        # Causal mask: upper triangle blocks future attention; padding mask blocks pad tokens
        pad_mask = mx.where(padding_mask[:, None, None, :], -1e9, 0.0)  # (B, 1, 1, L)
        causal = mx.triu(mx.full((L, L), -1e9), k=1)[None, None, :, :]  # (1, 1, L, L)
        attn_mask = pad_mask + causal
        h = self.encoder(h, mask=attn_mask)
        h = self.out_norm(h)
        return self.out_head(h)
