"""PyTorch mirror of LimbSequenceTransformer.

Same contract as ``mlx_architecture.LimbSequenceTransformer``:
  - input_dim: feature dimension per token (arrow features + chart-metadata + prev-limb one-hot)
  - d_model / n_heads / n_layers / ffn_dim / max_len / dropout / n_classes: identical semantics
  - causal upper-triangular attention + key-padding mask
  - sinusoidal positional encoding

Weight names DO NOT match the MLX implementation by design: PyTorch's
``nn.TransformerEncoderLayer`` uses fused QKV projections (``in_proj_weight``).
A converter script (``scripts/convert_weights_mlx_torch.py``) is the supported
path for cross-backend porting. Within a single backend the safetensors load
cleanly.

This module imports torch lazily so the package still works on machines
without a torch install — only training/inference on the torch backend pulls it.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError(
            'PyTorch backend requested but torch is not installed. '
            'Install via `pip install torch` (CPU/CUDA) or '
            '`pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` (AMD ROCm).'
        ) from e


def sinusoidal_pos_encoding(seq_len: int, d_model: int, device=None) -> "Tensor":
    torch, _ = _import_torch()
    pe = torch.zeros(seq_len, d_model, device=device)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    n_cos = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(positions * div_term[:n_cos])
    return pe


def build_model(
    input_dim: int,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    ffn_dim: int = 1024,
    max_len: int = 1024,
    dropout: float = 0.1,
    n_classes: int = 3,
):
    """Factory that returns an instance of LimbSequenceTransformerTorch.

    Lives outside the class so torch is only imported when this is called.
    """
    torch, nn = _import_torch()

    class OutputHead(nn.Module):
        def __init__(self, d: int, c: int, p: float):
            super().__init__()
            self.fc1 = nn.Linear(d, d // 2)
            self.act = nn.GELU()
            self.drop = nn.Dropout(p)
            self.fc2 = nn.Linear(d // 2, c)

        def forward(self, x):
            return self.fc2(self.drop(self.act(self.fc1(x))))

    class LimbSequenceTransformerTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.input_proj = nn.Linear(input_dim, d_model)
            self.register_buffer(
                'pos_enc',
                sinusoidal_pos_encoding(max_len, d_model),
                persistent=False,
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.dropout = nn.Dropout(dropout)
            self.out_norm = nn.LayerNorm(d_model)
            self.out_head = OutputHead(d_model, n_classes, dropout)
            self.d_model = d_model
            self.n_classes = n_classes
            self.input_dim = input_dim
            self.max_len = max_len

        def forward(self, x, padding_mask):
            """x: (B, L, D_in)  padding_mask: (B, L) bool, True = padding."""
            B, L, _ = x.shape
            h = self.input_norm(x)
            h = self.input_proj(h)
            h = h + self.pos_enc[:L].unsqueeze(0)
            h = self.dropout(h)
            causal = torch.zeros(L, L, device=x.device, dtype=h.dtype)
            causal.masked_fill_(
                torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1),
                float('-inf'),
            )
            h = self.encoder(h, mask=causal, src_key_padding_mask=padding_mask)
            h = self.out_norm(h)
            return self.out_head(h)

    return LimbSequenceTransformerTorch()


def pick_device(prefer: str | None = None):
    """Choose best available device.

    Order of preference (when ``prefer`` is None):
      1. CUDA / ROCm (both expose as ``torch.cuda``; ROCm needs
         ``HSA_OVERRIDE_GFX_VERSION=10.3.0`` for gfx1030/RDNA2 like the 6950 XT)
      2. MPS (Apple Silicon)
      3. CPU

    Pass ``prefer`` in {'cuda', 'mps', 'cpu'} to force a choice.
    """
    torch, _ = _import_torch()
    if prefer == 'cpu':
        return torch.device('cpu')
    if prefer == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA/ROCm not available')
        return torch.device('cuda')
    if prefer == 'mps':
        if not getattr(torch.backends, 'mps', None) or not torch.backends.mps.is_available():
            raise RuntimeError('MPS not available')
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_safetensors(model, path: str) -> None:
    """Save model weights as safetensors. Lazy-imports safetensors."""
    try:
        from safetensors.torch import save_file
    except ImportError as e:
        raise ImportError(
            'safetensors not installed. `pip install safetensors`.'
        ) from e
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, path)


def load_safetensors(model, path: str, device=None) -> None:
    """Load safetensors into an existing torch model in-place."""
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError('safetensors not installed. `pip install safetensors`.') from e
    state = load_file(path, device='cpu')
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
