#!/usr/bin/env python3
"""Convert PyTorch safetensors weights → MLX safetensors.

PyTorch TransformerEncoder uses fused in_proj_weight (QKV concatenated).
MLX MultiHeadAttention uses separate query_proj / key_proj / value_proj.

Run this on your Mac where MLX is available:

    python scripts/convert_torch_to_mlx.py \
        --torch_weights artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-torch-best.safetensors \
        --out           artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-mlx-best.safetensors \
        --sd            singles

    python scripts/convert_torch_to_mlx.py \
        --torch_weights artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-torch-best.safetensors \
        --out           artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-mlx-best.safetensors \
        --sd            doubles

After conversion, run compare_models.py or infer_v8_batch.py normally — the
MLX backend auto-loads *-mlx-best.safetensors.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np


def convert(torch_path: str, out_path: str) -> None:
    import safetensors.numpy as st_np

    # Load torch weights as numpy (safetensors are backend-agnostic)
    from safetensors import safe_open
    torch_weights: dict[str, np.ndarray] = {}
    with safe_open(torch_path, framework='numpy') as f:
        for k in f.keys():
            torch_weights[k] = f.get_tensor(k)

    print(f'Loaded {len(torch_weights)} tensors from {torch_path}')

    mlx_weights: dict[str, np.ndarray] = {}

    for k, v in torch_weights.items():
        # ── attention in_proj → separate Q / K / V ──────────────────────────
        if 'self_attn.in_proj_weight' in k:
            # PyTorch: (3*d_model, d_model) — Q, K, V stacked
            prefix = k.replace('self_attn.in_proj_weight', '')
            d = v.shape[0] // 3
            mlx_weights[prefix + 'attention.query_proj.weight'] = v[:d]
            mlx_weights[prefix + 'attention.key_proj.weight']   = v[d:2*d]
            mlx_weights[prefix + 'attention.value_proj.weight'] = v[2*d:]
            continue

        if 'self_attn.in_proj_bias' in k:
            prefix = k.replace('self_attn.in_proj_bias', '')
            d = v.shape[0] // 3
            mlx_weights[prefix + 'attention.query_proj.bias'] = v[:d]
            mlx_weights[prefix + 'attention.key_proj.bias']   = v[d:2*d]
            mlx_weights[prefix + 'attention.value_proj.bias'] = v[2*d:]
            continue

        if 'self_attn.out_proj' in k:
            new_k = k.replace('self_attn.out_proj', 'attention.out_proj')
            mlx_weights[new_k] = v
            continue

        # ── layer norms: norm1/norm2 stay the same ──────────────────────────
        # ── FFN: linear1/linear2 stay the same ──────────────────────────────
        # ── input_proj, output head, etc. ───────────────────────────────────
        mlx_weights[k] = v

    # Verify no NaN / Inf
    for k, v in mlx_weights.items():
        if not np.isfinite(v).all():
            print(f'WARNING: {k} contains non-finite values')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    st_np.save_file(mlx_weights, out_path)
    print(f'Saved {len(mlx_weights)} tensors → {out_path}')

    # Quick shape report
    n_attn = sum(1 for k in mlx_weights if 'query_proj' in k)
    print(f'  Attention splits: {n_attn} query_proj tensors')


def verify_mlx(out_path: str, sd: str) -> None:
    """Load converted weights into actual MLX model and check forward pass."""
    try:
        import mlx.core as mx
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer
        import json, os

        input_dim = 28 if sd == 'singles' else 33
        model_dir = str(Path(out_path).parent)
        config_path = os.path.join(model_dir, 'train_config.json')
        if os.path.exists(config_path):
            cfg = json.load(open(config_path))
        else:
            cfg = {}

        m = LimbSequenceTransformer(
            input_dim=input_dim,
            d_model=cfg.get('d_model', 384),
            n_heads=cfg.get('n_heads', 8),
            n_layers=cfg.get('n_layers', 8),
            ffn_dim=cfg.get('ffn_dim', 1536),
        )
        m.load_weights(out_path)
        mx.eval(m.parameters())

        # smoke test
        dummy_x    = mx.zeros((1, 16, input_dim))
        dummy_mask = mx.zeros((1, 16), dtype=mx.bool_)
        out = m(dummy_x, dummy_mask)
        mx.eval(out)
        print(f'MLX forward pass OK — output shape: {out.shape}')
    except ImportError:
        print('MLX not available — run this on your Mac to verify.')
    except Exception as e:
        print(f'MLX verify FAILED: {e}')
        print('Key name mismatch likely — check MLX weight names with:')
        print('  import mlx.nn as nn; m = nn.TransformerEncoder(...); print(list(m.flatten()[0].keys())[:10])')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--torch_weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sd', choices=['singles', 'doubles'], default='singles')
    p.add_argument('--verify', action='store_true', help='Test with MLX after conversion (Mac only)')
    a = p.parse_args()

    convert(a.torch_weights, a.out)
    if a.verify:
        verify_mlx(a.out, a.sd)


if __name__ == '__main__':
    main()
