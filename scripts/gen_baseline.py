#!/usr/bin/env python3
"""Generate a benchmark baseline JSON from available CSV chartstructs.

Samples a balanced set of singles+doubles charts across difficulty levels.
Output format matches what compare_models.py expects.

Usage:
    python scripts/gen_baseline.py \
        --csv_dir artifacts/manual-chartstructs/visss-120524 \
        --out artifacts/benchmark_baseline_local.json \
        --n 100
"""
from __future__ import annotations
import argparse
import json
import os
import re
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_chart_name(stem: str) -> dict | None:
    """Parse 'SongName_S15_ARCADE' or 'SongName_D22_ARCADE' → dict."""
    m = re.search(r'_([SD])(\d+)_', stem)
    if not m:
        return None
    mode = m.group(1)
    level = int(m.group(2))
    return {'shortname': stem, 'mode': mode, 'level': level}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=100, help='Total charts in baseline')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    all_csvs = sorted(Path(args.csv_dir).glob('*.csv'))
    entries = []
    for c in all_csvs:
        info = parse_chart_name(c.stem)
        if info:
            entries.append(info)

    singles = [e for e in entries if e['mode'] == 'S']
    doubles = [e for e in entries if e['mode'] == 'D']

    n_s = min(args.n // 2, len(singles))
    n_d = min(args.n - n_s, len(doubles))

    # stratify by level: pick uniformly across difficulty tiers
    def stratified_sample(pool, n):
        by_level = {}
        for e in pool:
            by_level.setdefault(e['level'], []).append(e)
        levels = sorted(by_level)
        chosen = []
        per_level = max(1, n // len(levels))
        for lv in levels:
            chosen.extend(random.sample(by_level[lv], min(per_level, len(by_level[lv]))))
        if len(chosen) < n:
            remaining = [e for e in pool if e not in chosen]
            chosen.extend(random.sample(remaining, min(n - len(chosen), len(remaining))))
        return random.sample(chosen, min(n, len(chosen)))

    selected = stratified_sample(singles, n_s) + stratified_sample(doubles, n_d)
    random.shuffle(selected)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(selected, f, indent=2)

    print(f'Wrote {len(selected)} entries → {args.out}')
    s_cnt = sum(1 for e in selected if e['mode'] == 'S')
    d_cnt = sum(1 for e in selected if e['mode'] == 'D')
    levels = sorted(set(e['level'] for e in selected))
    print(f'  Singles: {s_cnt}  Doubles: {d_cnt}  Levels: {levels[0]}–{levels[-1]}')


if __name__ == '__main__':
    main()
