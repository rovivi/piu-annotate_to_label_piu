#!/usr/bin/env python3
"""
Hard-negative mining for MLX limb prediction models.

Runs a trained model over the training dataset, identifies arrows where
predictions disagree with ground truth, and writes an index file for
focused fine-tuning.

Usage:
    python scripts/hard_negative_mining.py \
        --model_dir artifacts/models/visss-mlx-v9 \
        --cache_dir artifacts/cache/mlx-v9-singles \
        --out_path artifacts/hard_negatives/singles_hard_negatives.json \
        --singles_or_doubles singles \
        --batch_size 16
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

INT_TO_LIMB = {0: 'L', 1: 'R', 2: 'E'}


def load_model(model_dir: str, sd: str) -> tuple[LimbSequenceTransformer, dict]:
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
    logger.info(f'Loaded model from {weights_path}')
    return model, cfg


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


def model_forward(model: LimbSequenceTransformer, x_chunk: np.ndarray) -> np.ndarray:
    """Run a single chunk through the model, returns softmax probs (L, 3)."""
    L = len(x_chunk)
    x_mx = mx.array(x_chunk[None])
    pm = mx.zeros((1, L), dtype=mx.bool_)
    logits = model(x_mx, pm)
    mx.eval(logits)
    probs = np.array(mx.softmax(logits, axis=-1))[0]
    return probs


def predict_sequence(model: LimbSequenceTransformer, x_full: np.ndarray) -> np.ndarray:
    """Predict over a full sequence with overlapping chunks; average overlaps."""
    N = len(x_full)
    prob_acc = np.zeros((N, 3), dtype=np.float64)
    weight_acc = np.zeros(N, dtype=np.float64)
    for chunk, slc in make_chunks(x_full):
        probs = model_forward(model, chunk)
        chunk_len = slc.stop - slc.start
        w = np.ones(chunk_len, dtype=np.float64)
        if chunk_len > 2 * CHUNK_OVERLAP:
            ramp = np.linspace(0.3, 1.0, CHUNK_OVERLAP)
            w[:CHUNK_OVERLAP] = ramp
            w[-CHUNK_OVERLAP:] = ramp[::-1]
        prob_acc[slc] += probs * w[:, None]
        weight_acc[slc] += w
    weight_acc = np.maximum(weight_acc, 1e-8)
    return np.argmax(prob_acc / weight_acc[:, None], axis=-1).astype(np.int32)


def ar_infer(model: LimbSequenceTransformer, x_base: np.ndarray) -> np.ndarray:
    """2-pass autoregressive inference."""
    N = len(x_base)
    prev1 = np.zeros((N, 4), dtype=np.float32)
    prev1[:, 3] = 0.0
    prev1[0, 3] = 1.0
    x1 = np.concatenate([x_base, prev1], axis=1)
    preds1 = predict_sequence(model, x1)

    prev2 = _prev_limb_onehot_from_preds(preds1, first_label=3)
    x2 = np.concatenate([x_base, prev2], axis=1)
    preds2 = predict_sequence(model, x2)
    return preds2


def build_chunks_with_prev(x, y, max_len=MAX_SEQ_LEN, overlap=CHUNK_OVERLAP):
    """Same logic as train_mlx.py for validation."""
    if len(x) <= max_len:
        prev = np.zeros((len(y), 4), dtype=np.float32)
        prev[0, 3] = 1.0
        for i in range(1, len(y)):
            lbl = int(y[i - 1])
            prev[i, min(lbl, 2)] = 1.0
        return [(np.concatenate([x, prev], axis=1), y)]

    chunks = []
    stride = max_len - overlap
    for start in range(0, len(x) - max_len + 1, stride):
        cx = x[start:start + max_len]
        cy = y[start:start + max_len]
        first = 3 if start == 0 else min(int(y[start - 1]), 2)
        prev = np.zeros((len(cy), 4), dtype=np.float32)
        prev[0, first] = 1.0
        for i in range(1, len(cy)):
            lbl = int(cy[i - 1])
            prev[i, min(lbl, 2)] = 1.0
        chunks.append((np.concatenate([cx, prev], axis=1), cy))

    if (len(x) - max_len) % stride != 0:
        start = len(x) - max_len
        cx = x[start:]
        cy = y[start:]
        first = 3 if start == 0 else min(int(y[start - 1]), 2)
        prev = np.zeros((len(cy), 4), dtype=np.float32)
        prev[0, first] = 1.0
        for i in range(1, len(cy)):
            lbl = int(cy[i - 1])
            prev[i, min(lbl, 2)] = 1.0
        chunks.append((np.concatenate([cx, prev], axis=1), cy))

    return chunks


def analyze_errors(model: LimbSequenceTransformer, cache_dir: str, sd: str, batch_size: int):
    all_files = sorted(Path(cache_dir).glob('*.npz'))
    logger.info(f'Analyzing {len(all_files)} files from {cache_dir}')

    hard_negatives = []  # list of dicts
    per_class_errors = {0: 0, 1: 0, 2: 0}
    per_class_total = {0: 0, 1: 0, 2: 0}
    total_errors = 0
    total_tokens = 0

    for npz_path in tqdm(all_files, desc='Mining hard negatives'):
        d = np.load(npz_path)
        x = d['x'].astype(np.float32)
        y = d['y'].astype(np.int32)

        # AR inference
        preds = ar_infer(model, x)

        # Find errors
        valid = y >= 0
        errors = (preds != y) & valid
        total_errors += errors.sum()
        total_tokens += valid.sum()

        for c in range(3):
            mask = valid & (y == c)
            per_class_total[c] += mask.sum()
            per_class_errors[c] += (errors[mask]).sum()

        if errors.sum() > 0:
            error_indices = np.where(errors)[0].tolist()
            # Confidence of wrong predictions
            # We need to get probs for error positions
            # Re-run inference to get full-sequence probs
            # Simplification: just use the preds array
            hard_negatives.append({
                'file': str(npz_path.name),
                'n_arrows': int(len(y)),
                'n_errors': int(errors.sum()),
                'error_rate': float(errors.sum() / valid.sum()),
                'error_indices': error_indices,
                'pred_labels': preds[error_indices].tolist(),
                'gt_labels': y[error_indices].tolist(),
            })

    overall_acc = 1.0 - total_errors / max(total_tokens, 1)
    logger.info(f'Overall AR accuracy on training set: {overall_acc*100:.2f}%')
    logger.info(f'Total errors: {total_errors} / {total_tokens}')
    for c in range(3):
        cls_acc = 1.0 - per_class_errors[c] / max(per_class_total[c], 1)
        logger.info(f'  Class {INT_TO_LIMB[c]}: {cls_acc*100:.1f}% ({per_class_total[c] - per_class_errors[c]}/{per_class_total[c]})')

    return hard_negatives, overall_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--singles_or_doubles', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    model, cfg = load_model(args.model_dir, args.singles_or_doubles)
    hard_negatives, overall_acc = analyze_errors(
        model, args.cache_dir, args.singles_or_doubles, args.batch_size
    )

    # Sort by error rate descending
    hard_negatives.sort(key=lambda x: x['error_rate'], reverse=True)

    output = {
        'model_dir': args.model_dir,
        'cache_dir': args.cache_dir,
        'singles_or_doubles': args.singles_or_doubles,
        'overall_ar_accuracy': overall_acc,
        'n_charts_analyzed': len(hard_negatives),
        'n_hard_negative_charts': len([h for h in hard_negatives if h['n_errors'] > 0]),
        'hard_negatives': hard_negatives,
    }

    with open(args.out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.success(f'Saved hard-negative index to {args.out_path}')
    logger.info(f'Top 10 worst charts by error rate:')
    for h in hard_negatives[:10]:
        logger.info(f"  {h['file']}: {h['error_rate']*100:.1f}% ({h['n_errors']}/{h['n_arrows']})")


if __name__ == '__main__':
    main()
