#!/usr/bin/env python3
"""Train LightGBM match_next and match_prev models from the NPZ cache.

These are binary classifiers: "does this arrow use the same foot as the
next/prev arrow?" They plug into the Tactician score function (terms 2 and 3),
which are currently no-ops (DummyMatchModel) for transformer backends.

With real match models the beam search and flip_labels_by_score can
differentiate candidates on 3/3 score terms instead of 1/3.

Output: <out_dir>/<sd>-arrows_to_matchnext-lgbm.txt
        <out_dir>/<sd>-arrows_to_matchprev-lgbm.txt

ModelSuite will auto-load these if they exist alongside the main weights.

Usage:
    python scripts/train_match_models.py --sd singles \
        --cache_dir artifacts/cache/torch-singles \
        --out_dir  artifacts/models/visss-torch-v9-singles

    python scripts/train_match_models.py --sd doubles \
        --cache_dir artifacts/cache/torch-doubles \
        --out_dir  artifacts/models/visss-torch-v9-doubles
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def build_match_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (feat_next, feat_prev, idx_next, idx_prev) from raw x (N, D).

    For match_next at position i: concatenate x[i] + x[i+1].
    For match_prev at position i: concatenate x[i] + x[i-1].
    Position 0 is excluded from match_prev; position N-1 from match_next.
    """
    n = len(x)
    # match_next: positions 0..N-2
    feat_next = np.concatenate([x[:-1], x[1:]], axis=1)
    idx_next = np.arange(n - 1)
    # match_prev: positions 1..N-1
    feat_prev = np.concatenate([x[1:], x[:-1]], axis=1)
    idx_prev = np.arange(1, n)
    return feat_next, feat_prev, idx_next, idx_prev


def load_dataset(cache_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_next, y_next, X_prev, y_prev) across all npz files.
    Only uses arrows with limb label 0 (L) or 1 (R); skips 'either' (2).
    """
    files = sorted(Path(cache_dir).glob('*.npz'))
    logger.info(f'Loading {len(files)} files from {cache_dir}')

    all_feat_next, all_y_next = [], []
    all_feat_prev, all_y_prev = [], []

    for f in files:
        d = np.load(f)
        x = d['x'].astype(np.float32)
        y = d['y'].astype(np.int8)
        n = len(x)
        if n < 2:
            continue

        feat_next, feat_prev, idx_next, idx_prev = build_match_features(x)

        # match_next labels: 1 if same foot, 0 if different
        mn_labels = (y[idx_next] == y[idx_next + 1]).astype(np.int8)
        # keep only rows where both i and i+1 are L or R (not E=2)
        valid_n = (y[idx_next] < 2) & (y[idx_next + 1] < 2)
        all_feat_next.append(feat_next[valid_n])
        all_y_next.append(mn_labels[valid_n])

        # match_prev labels: 1 if same foot, 0 if different
        mp_labels = (y[idx_prev] == y[idx_prev - 1]).astype(np.int8)
        valid_p = (y[idx_prev] < 2) & (y[idx_prev - 1] < 2)
        all_feat_prev.append(feat_prev[valid_p])
        all_y_prev.append(mp_labels[valid_p])

        # also add augmented mirror (x_mirror, y_mirror if present)
        if 'x_mirror' in d and 'y_mirror' in d:
            xm = d['x_mirror'].astype(np.float32)
            ym = d['y_mirror'].astype(np.int8)
            if len(xm) >= 2:
                fm_n, fm_p, im_n, im_p = build_match_features(xm)
                mn_m = (ym[im_n] == ym[im_n + 1]).astype(np.int8)
                vm_n = (ym[im_n] < 2) & (ym[im_n + 1] < 2)
                all_feat_next.append(fm_n[vm_n])
                all_y_next.append(mn_m[vm_n])
                mp_m = (ym[im_p] == ym[im_p - 1]).astype(np.int8)
                vm_p = (ym[im_p] < 2) & (ym[im_p - 1] < 2)
                all_feat_prev.append(fm_p[vm_p])
                all_y_prev.append(mp_m[vm_p])

    X_next = np.concatenate(all_feat_next, axis=0)
    y_next = np.concatenate(all_y_next, axis=0)
    X_prev = np.concatenate(all_feat_prev, axis=0)
    y_prev = np.concatenate(all_y_prev, axis=0)
    logger.info(f'match_next: {len(X_next)} samples, class balance: {y_next.mean():.3f}')
    logger.info(f'match_prev: {len(X_prev)} samples, class balance: {y_prev.mean():.3f}')
    return X_next, y_next, X_prev, y_prev


def train_lgbm(X: np.ndarray, y: np.ndarray, task: str) -> lgb.Booster:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    tr_ds = lgb.Dataset(X_tr, label=y_tr)
    val_ds = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective':       'binary',
        'metric':          'binary_logloss',
        'num_leaves':      63,
        'learning_rate':   0.05,
        'n_estimators':    300,
        'min_child_samples': 20,
        'subsample':       0.8,
        'colsample_bytree': 0.8,
        'verbose':         -1,
    }
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=True),
                 lgb.log_evaluation(period=50)]
    bst = lgb.train(
        params, tr_ds,
        num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=callbacks,
    )
    preds = bst.predict(X_val)
    acc = accuracy_score(y_val, (preds >= 0.5).astype(int))
    auc = roc_auc_score(y_val, preds)
    logger.success(f'{task}: val_acc={acc*100:.2f}%  AUC={auc:.4f}  trees={bst.num_trees()}')
    return bst


def main():
    p = argparse.ArgumentParser(description='Train match_next/match_prev LightGBM models')
    p.add_argument('--sd', required=True, choices=['singles', 'doubles'])
    p.add_argument('--cache_dir', required=True, help='Directory of .npz cache files')
    p.add_argument('--out_dir', required=True, help='Directory to save model .txt files')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_next, y_next, X_prev, y_prev = load_dataset(args.cache_dir)

    logger.info('Training match_next model...')
    bst_next = train_lgbm(X_next, y_next, task='match_next')
    path_next = os.path.join(args.out_dir, f'{args.sd}-arrows_to_matchnext-lgbm.txt')
    bst_next.save_model(path_next)
    logger.success(f'Saved: {path_next}')

    logger.info('Training match_prev model...')
    bst_prev = train_lgbm(X_prev, y_prev, task='match_prev')
    path_prev = os.path.join(args.out_dir, f'{args.sd}-arrows_to_matchprev-lgbm.txt')
    bst_prev.save_model(path_prev)
    logger.success(f'Saved: {path_prev}')


if __name__ == '__main__':
    main()
