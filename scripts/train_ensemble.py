#!/usr/bin/env python3
"""
Train an ensemble of MLX models with different seeds.
Launches multiple training runs and optionally evaluates ensemble accuracy.

Usage:
    python scripts/train_ensemble.py \
        --cache_dir artifacts/cache/mlx-v9-singles \
        --out_dir artifacts/models/visss-mlx-v9-ensemble \
        --singles_or_doubles singles \
        --n_models 3 \
        --epochs 40 --batch_size 16 --lr 3e-4 \
        --ss_prob_max 0.5 --large_model
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--singles_or_doubles', type=str, required=True)
    parser.add_argument('--n_models', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ss_prob_max', type=float, default=0.5)
    parser.add_argument('--large_model', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without running')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seeds = list(range(args.n_models))
    processes = []

    for seed in seeds:
        model_out = os.path.join(args.out_dir, f'seed{seed}')
        os.makedirs(model_out, exist_ok=True)

        cmd = [
            'python3', 'cli/limbuse/train_mlx.py',
            '--singles_or_doubles', args.singles_or_doubles,
            '--manual_chart_struct_folder', args.cache_dir,
            '--out_dir', model_out,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--seed', str(seed),
            '--ss_prob_max', str(args.ss_prob_max),
        ]
        if args.large_model:
            cmd.append('--large_model')

        log_path = os.path.join(model_out, 'train.log')

        if args.dry_run:
            logger.info(f'[DRY RUN] Seed {seed}: {" ".join(cmd)} > {log_path} 2>&1')
            continue

        logger.info(f'Launching seed {seed} → {model_out}')
        with open(log_path, 'w') as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            processes.append((seed, p, log_path))

    if args.dry_run:
        return

    logger.info(f'Launched {len(processes)} training jobs. Waiting for completion...')
    for seed, p, log_path in processes:
        exit_code = p.wait()
        if exit_code == 0:
            logger.success(f'Seed {seed} completed successfully')
        else:
            logger.error(f'Seed {seed} failed with exit code {exit_code}. Log: {log_path}')

    # Summarize results
    logger.info('=== Ensemble Summary ===')
    for seed in seeds:
        model_out = os.path.join(args.out_dir, f'seed{seed}')
        config_path = os.path.join(model_out, 'train_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            logger.info(f"  Seed {seed}: best_val_acc={cfg.get('best_acc', 0)*100:.1f}%")
        else:
            logger.warning(f"  Seed {seed}: no config found")


if __name__ == '__main__':
    main()
