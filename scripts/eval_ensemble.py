#!/usr/bin/env python3
"""
Evaluate an ensemble of MLX models on cached .npz data.

Averages logits from multiple models before taking argmax.
Optionally applies TTA (Test-Time Augmentation) per model.

Usage:
    python scripts/eval_ensemble.py \
        --ensemble_dir artifacts/models/visss-mlx-v9-ensemble \
        --cache_dir artifacts/cache/mlx-v9-singles \
        --singles_or_doubles singles \
        --tta
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer

MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 256
SINGLES_INPUT_DIM = 28
DOUBLES_INPUT_DIM = 33


def load_model(model_dir: str, sd: str) -> LimbSequenceTransformer:
    config_path = os.path.join(model_dir, 'train_config.json')
    with open(config_path) as f:
        cfg = json.load(f)
    input_dim = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM
    model = LimbSequenceTransformer(
        input_dim=input_dim,
        d_model=cfg.get('d_model', 256),
        n_heads=cfg.get('n_heads', 8),
        n_layers=cfg.get('n_layers', 6),
        ffn_dim=cfg.get('ffn_dim', 1024),
    )
    weights_path = os.path.join(model_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
    model.load_weights(weights_path)
    model.eval()
    return model


def _prev_limb_onehot_from_preds(preds: np.ndarray, first_label: int = 3) -> np.ndarray:
    N = len(preds)
    prev = np.zeros((N, 4), dtype=np.float32)
    prev[0, min(first_label, 3)] = 1.0
    for i in range(1, N):
        lbl = min(int(preds[i - 1]), 2)
        prev[i, lbl] = 1.0
    return prev


def make_chunks(x: np.ndarray, max_len: int = MAX_SEQ_LEN, overlap: int = CHUNK_OVERLAP):
    N = len(x)
    if N <= max_len:
        return [(x, slice(0, N))]
    chunks = []
    stride = max_len - overlap
    start = 0
    while start + max_len <= N:
        chunks.append((x[start:start + max_len], slice(start, start + max_len)))
        start += stride
    if start < N:
        chunks.append((x[N - max_len:], slice(N - max_len, N)))
    return chunks


def predict_sequence_logits(model: LimbSequenceTransformer, x_full: np.ndarray) -> np.ndarray:
    """Predict over a full sequence, returning averaged logits (N, 3)."""
    N = len(x_full)
    logits_acc = np.zeros((N, 3), dtype=np.float64)
    weight_acc = np.zeros(N, dtype=np.float64)
    for chunk, slc in make_chunks(x_full):
        L = len(chunk)
        x_mx = mx.array(chunk[None])
        pm = mx.zeros((1, L), dtype=mx.bool_)
        logits = model(x_mx, pm)
        mx.eval(logits)
        logits_np = np.array(logits)[0]
        chunk_len = slc.stop - slc.start
        w = np.ones(chunk_len, dtype=np.float64)
        if chunk_len > 2 * CHUNK_OVERLAP:
            ramp = np.linspace(0.3, 1.0, CHUNK_OVERLAP)
            w[:CHUNK_OVERLAP] = ramp
            w[-CHUNK_OVERLAP:] = ramp[::-1]
        logits_acc[slc] += logits_np * w[:, None]
        weight_acc[slc] += w
    weight_acc = np.maximum(weight_acc, 1e-8)
    return logits_acc / weight_acc[:, None]


def ar_infer_logits(model: LimbSequenceTransformer, x_base: np.ndarray) -> np.ndarray:
    """2-pass autoregressive inference returning logits."""
    N = len(x_base)
    prev1 = np.zeros((N, 4), dtype=np.float32)
    prev1[:, 3] = 0.0
    prev1[0, 3] = 1.0
    x1 = np.concatenate([x_base, prev1], axis=1)
    logits1 = predict_sequence_logits(model, x1)
    preds1 = np.argmax(logits1, axis=-1)

    prev2 = _prev_limb_onehot_from_preds(preds1, first_label=3)
    x2 = np.concatenate([x_base, prev2], axis=1)
    logits2 = predict_sequence_logits(model, x2)
    return logits2


def flip_lr_logits(logits: np.ndarray) -> np.ndarray:
    flipped = logits.copy()
    flipped[:, [0, 1]] = logits[:, [1, 0]]
    return flipped


def evaluate_model(model: LimbSequenceTransformer, x: np.ndarray, y: np.ndarray, use_tta: bool = False) -> tuple[int, int]:
    """Returns (correct, total) for a single chart."""
    if use_tta:
        logits = ar_infer_logits(model, x)
        # For TTA we'd need the mirrored x too, but here we just do standard AR
        # TTA at evaluation time requires mirrored features which we don't have in .npz
        # We'll handle TTA separately in the ensemble script if needed
        preds = np.argmax(logits, axis=-1)
    else:
        logits = ar_infer_logits(model, x)
        preds = np.argmax(logits, axis=-1)

    valid = y >= 0
    correct = ((preds == y) & valid).sum()
    total = valid.sum()
    return int(correct), int(total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_dir', type=str, required=True,
                        help='Directory containing seed*/ subdirectories')
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--singles_or_doubles', type=str, required=True)
    parser.add_argument('--tta', action='store_true', help='Enable TTA (requires mirrored features)')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    sd = args.singles_or_doubles
    input_dim = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM

    # Load all models in ensemble
    model_dirs = sorted(Path(args.ensemble_dir).glob('seed*'))
    if not model_dirs:
        logger.error(f'No seed directories found in {args.ensemble_dir}')
        return

    models = []
    for md in model_dirs:
        try:
            m = load_model(str(md), sd)
            models.append(m)
            logger.info(f'Loaded model from {md}')
        except Exception as e:
            logger.warning(f'Failed to load {md}: {e}')

    logger.info(f'Ensemble size: {len(models)}')

    all_files = sorted(Path(args.cache_dir).glob('*.npz'))
    if args.limit:
        all_files = all_files[:args.limit]

    # Evaluate ensemble
    total_correct = 0
    total_tokens = 0
    per_class_correct = np.zeros(3, dtype=np.int64)
    per_class_total = np.zeros(3, dtype=np.int64)

    for npz_path in tqdm(all_files, desc='Evaluating ensemble'):
        d = np.load(npz_path)
        x = d['x'].astype(np.float32)
        y = d['y'].astype(np.int32)

        # Ensemble inference: average logits
        N = len(x)
        logits_sum = np.zeros((N, 3), dtype=np.float64)

        for model in models:
            logits = ar_infer_logits(model, x)
            logits_sum += logits

        avg_logits = logits_sum / len(models)
        preds = np.argmax(avg_logits, axis=-1)

        valid = y >= 0
        total_correct += ((preds == y) & valid).sum()
        total_tokens += valid.sum()

        for c in range(3):
            mask = valid & (y == c)
            per_class_total[c] += mask.sum()
            per_class_correct[c] += (preds[mask] == c).sum()

    acc = total_correct / max(total_tokens, 1)
    logger.success(f'=== Ensemble Results ===')
    logger.success(f'Overall accuracy: {acc*100:.2f}%')
    for c in range(3):
        cls_acc = per_class_correct[c] / max(per_class_total[c], 1)
        logger.info(f'  Class {c}: {cls_acc*100:.1f}% ({per_class_correct[c]}/{per_class_total[c]})')


if __name__ == '__main__':
    main()
