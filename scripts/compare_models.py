from __future__ import annotations
"""compare_models.py — multi-model evaluation against vis-ss ground truth.

Runs inference with N model configurations on a baseline chart set and prints
+ writes JSON tables with:
    - Per-pattern accuracy (tap / jack / triple / bracket_LL / bracket_RR /
      bracket_LR / jump / hold_release / stream)
    - Confusion matrix L/R/E
    - Per-difficulty-level accuracy
    - Per-chart summary

Usage examples
--------------
    # Single model, legacy-style
    python scripts/compare_models.py --models lgbm:artifacts/models/visss

    # Three models, plot + JSON output
    python scripts/compare_models.py \\
        --models lgbm:artifacts/models/visss \\
        --models mlx_v8:artifacts/models/visss-mlx-v8 \\
        --models torch:artifacts/models/visss-torch \\
        --baseline artifacts/benchmark_baseline_60.json \\
        --output_json artifacts/comparator/multi_compare.json \\
        --plot artifacts/comparator/multi_compare.png

Each ``--models`` value is ``name:dir`` where the directory is fed into
``hargs['model.dir']``. The backend is auto-detected by looking at filenames:

    - ``*-lightgbm-best.safetensors`` or ``*.txt`` → ``lightgbm``
    - ``*-mlx-best.safetensors``                  → ``mlx``
    - ``*-torch-best.safetensors``                → ``torch``

Override with ``name:dir:backend`` if needed.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hackerargs import args as hargs

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml import predictor as ml_predictor


VIS_DIR = '/Users/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524'
CSV_DIR = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524'
BASELINE = 'artifacts/benchmark_baseline_60.json'


L2I = {'l': 0, 'r': 1, 'e': 2, 'h': 1, '?': -1}
I2L = {0: 'l', 1: 'r', 2: 'e'}


# ---------------------------------------------------------------------------
# Backend autodetect + setup
# ---------------------------------------------------------------------------


def detect_backend(model_dir: str) -> str:
    """Sniff filenames to pick a backend."""
    files = os.listdir(model_dir) if os.path.isdir(model_dir) else []
    if any('-mlx-' in f for f in files):
        return 'mlx'
    if any('-torch-' in f for f in files):
        return 'torch'
    if any(f.endswith('.txt') for f in files):
        return 'lightgbm'
    raise RuntimeError(f'Could not auto-detect backend in {model_dir}; files: {files[:10]}')


def setup_model_args(sd: str, model_dir: str, backend: str):
    hargs['model'] = backend
    hargs['model.dir'] = model_dir
    if backend == 'lightgbm':
        hargs[f'model.arrows_to_limb-{sd}']      = f'{sd}-arrows_to_limb.txt'
        hargs[f'model.arrowlimbs_to_limb-{sd}']  = f'{sd}-arrowlimbs_to_limb.txt'
        hargs[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-arrows_to_matchnext.txt'
        hargs[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-arrows_to_matchprev.txt'
    else:
        # Transformer backends use canonical names; ModelSuite resolves them.
        hargs[f'model.arrows_to_limb-{sd}']      = f'{sd}-arrows_to_limb-{backend}-best'
        hargs[f'model.arrowlimbs_to_limb-{sd}']  = f'{sd}-arrowlimbs_to_limb-{backend}-best'


# ---------------------------------------------------------------------------
# Pattern bucketing
# ---------------------------------------------------------------------------


PATTERN_BUCKETS = (
    'tap', 'jack', 'triple', 'jump',
    'bracket_ll', 'bracket_rr', 'bracket_lr',
    'hold_release', 'stream',
)


def classify_pattern(
    pc,
    cs: ChartStruct,
    fcs,
    gt_limb_char: str,
    time_count: dict[float, int],
    panel_to_last_t: dict[int, float],
) -> list[str]:
    """Return all bucket tags that apply to this (pred_coord, ground-truth) pair."""
    row = cs.df.iloc[pc.row_idx]
    line = str(row['Line with active holds']).replace('`', '')
    n_dp = sum(1 for ch in line if ch in '12')
    t = round(float(row['Time']), 5)

    tags = []
    if n_dp == 1:
        tags.append('tap')
        prev_t = panel_to_last_t.get(pc.arrow_pos)
        if prev_t is not None and abs(t - prev_t) > 1e-3:
            tags.append('jack')
        if 0 < t - panel_to_last_t.get(-1, t - 1e9) < 0.12:
            tags.append('stream')
    elif n_dp == 2:
        # Bracket-able lookup based on featurized field — simpler than re-deriving.
        bracketable = bool(row.get('line_is_bracketable', False)) if 'line_is_bracketable' in cs.df.columns else False
        if bracketable:
            if gt_limb_char == 'l':
                tags.append('bracket_ll')
            elif gt_limb_char == 'r':
                tags.append('bracket_rr')
            else:
                tags.append('bracket_lr')
        else:
            tags.append('jump')
    elif time_count.get(t, 0) >= 3:
        tags.append('triple')

    # Hold release detection — only true if the line is purely a release (3s only)
    has_three = '3' in line
    has_one_two = any(ch in '12' for ch in line)
    if has_three and not has_one_two:
        tags.append('hold_release')
    return tags


# ---------------------------------------------------------------------------
# Per-model inference + scoring
# ---------------------------------------------------------------------------


def run_inference(csv_path: str, suite: ModelSuite) -> tuple | None:
    """Returns (pred_limbs, labels_gt, pred_coords, fcs, cs) or None."""
    try:
        cs = ChartStruct.from_file(csv_path)
        _cs, fcs, pred_limbs = ml_predictor.predict(cs, suite)
        labels = fcs.get_labels_from_limb_col('Limb annotation')
        return pred_limbs, labels, fcs.pred_coords, fcs, cs
    except Exception as e:
        logger.warning(f'Inference failed for {csv_path}: {e}')
        return None


def score_chart(pred_limbs, labels, pred_coords, fcs, cs) -> dict:
    """Per-pattern + confusion stats for a single chart."""
    pattern_total = defaultdict(int)
    pattern_correct = defaultdict(int)
    confusion = np.zeros((3, 3), dtype=int)  # rows=truth, cols=pred

    panel_to_last_t = {}
    time_count = defaultdict(int)
    for pc in pred_coords:
        t = round(float(cs.df.at[pc.row_idx, 'Time']), 5)
        time_count[t] += 1

    last_any_t = -1e9
    for idx, pc in enumerate(pred_coords):
        p = int(pred_limbs[idx])
        g = int(labels[idx])
        if 0 <= g <= 2:
            confusion[g, p] += 1
        gt_char = I2L.get(g, '?')
        tags = classify_pattern(
            pc, cs, fcs, gt_char, time_count, {**panel_to_last_t, -1: last_any_t},
        )
        t = round(float(cs.df.at[pc.row_idx, 'Time']), 5)
        correct = int(p == g)
        pattern_total['all'] += 1
        pattern_correct['all'] += correct
        for tag in tags:
            pattern_total[tag] += 1
            pattern_correct[tag] += correct
        panel_to_last_t[pc.arrow_pos] = t
        last_any_t = t

    return {
        'pattern_total': dict(pattern_total),
        'pattern_correct': dict(pattern_correct),
        'confusion': confusion.tolist(),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(per_chart: list[dict]) -> dict:
    """Sum per-chart stats into a global summary."""
    pt, pc_ = defaultdict(int), defaultdict(int)
    conf = np.zeros((3, 3), dtype=int)
    for r in per_chart:
        for k, v in r.get('pattern_total', {}).items():
            pt[k] += v
        for k, v in r.get('pattern_correct', {}).items():
            pc_[k] += v
        conf += np.array(r.get('confusion', [[0]*3]*3))
    pattern_acc = {k: pc_[k] / max(pt[k], 1) for k in pt}
    return {
        'pattern_total': dict(pt),
        'pattern_correct': dict(pc_),
        'pattern_acc': pattern_acc,
        'confusion': conf.tolist(),
    }


def per_level(per_chart_with_meta: list[dict]) -> dict:
    """Bucket overall accuracy by chart level."""
    bucket = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in per_chart_with_meta:
        lv = r['level']
        bucket[lv]['correct'] += r['pattern_correct'].get('all', 0)
        bucket[lv]['total']   += r['pattern_total'].get('all', 0)
    return {
        str(lv): bucket[lv]['correct'] / max(bucket[lv]['total'], 1)
        for lv in sorted(bucket)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_model_arg(s: str) -> tuple[str, str, str | None]:
    parts = s.split(':')
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f'--models expects name:dir[:backend], got {s!r}')


def make_confusion_plot(matrices: dict[str, np.ndarray], out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, matrices.items()):
        m = m.astype(float)
        row_sums = m.sum(axis=1, keepdims=True)
        m_norm = np.divide(m, np.maximum(row_sums, 1), where=row_sums > 0)
        im = ax.imshow(m_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['L', 'R', 'E'])
        ax.set_yticklabels(['L', 'R', 'E'])
        ax.set_xlabel('Pred'); ax.set_ylabel('GT')
        ax.set_title(name, fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{m_norm[i, j]*100:.1f}%\n({int(m[i, j])})',
                        ha='center', va='center',
                        color='white' if m_norm[i, j] > 0.5 else 'black',
                        fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    logger.success(f'Confusion plot saved: {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', action='append', required=True,
                   help='name:dir[:backend], repeatable')
    p.add_argument('--baseline', default=BASELINE)
    p.add_argument('--csv_dir', default=CSV_DIR)
    p.add_argument('--vis_dir', default=VIS_DIR)
    p.add_argument('--output_json', default=None)
    p.add_argument('--plot', default=None,
                   help='Path to save confusion matrix plot.')
    p.add_argument('--limit', type=int, default=None,
                   help='Limit baseline charts (smoke test).')
    a = p.parse_args()

    baseline = json.load(open(a.baseline))
    if a.limit:
        baseline = baseline[:a.limit]

    parsed_models = [parse_model_arg(s) for s in a.models]
    logger.info(f'Comparing {len(parsed_models)} models on {len(baseline)} charts')

    # Pre-load suites for each (model × sd) — heavy.
    suites: dict[tuple[str, str], ModelSuite] = {}
    for name, mdir, backend_override in parsed_models:
        backend = backend_override or detect_backend(mdir)
        logger.info(f'Loading {name} (backend={backend}) from {mdir}')
        for sd in ('singles', 'doubles'):
            setup_model_args(sd, mdir, backend)
            try:
                suites[(name, sd)] = ModelSuite(sd, backend=backend)
            except Exception as e:
                logger.warning(f'{name}/{sd} failed to load: {e}')

    per_model_results: dict[str, list[dict]] = {name: [] for name, _, _ in parsed_models}

    for entry in baseline:
        sn = entry['shortname']
        sd = 'singles' if entry['mode'] == 'S' else 'doubles'
        csv_path = os.path.join(a.csv_dir, sn + '.csv')
        if not os.path.exists(csv_path):
            logger.warning(f'No CSV: {sn}')
            continue
        for name, _, _ in parsed_models:
            suite = suites.get((name, sd))
            if suite is None:
                continue
            res = run_inference(csv_path, suite)
            if res is None:
                continue
            pred_limbs, labels, pred_coords, fcs, cs = res
            stats = score_chart(pred_limbs, labels, pred_coords, fcs, cs)
            stats.update({
                'shortname': sn,
                'song_name': entry.get('song_name'),
                'mode': entry.get('mode'),
                'level': entry.get('level'),
                'groups': entry.get('groups', []),
            })
            per_model_results[name].append(stats)

    # Aggregate
    summary = {}
    confusion_matrices = {}
    for name, _, _ in parsed_models:
        rows = per_model_results[name]
        agg = aggregate(rows)
        summary[name] = {
            'pattern_acc': agg['pattern_acc'],
            'pattern_total': agg['pattern_total'],
            'confusion': agg['confusion'],
            'per_level': per_level(rows),
        }
        confusion_matrices[name] = np.array(agg['confusion'])

    # Print
    all_patterns = ['all'] + list(PATTERN_BUCKETS)
    width = max(12, max(len(n) for n, _, _ in parsed_models))
    print('\n=== Pattern accuracy ===\n')
    header = f'{"pattern":<14}  {"n":>8}  ' + '  '.join(f'{n:>{width}}' for n, _, _ in parsed_models)
    print(header)
    print('-' * len(header))
    for pat in all_patterns:
        ns = [summary[n]['pattern_total'].get(pat, 0) for n, _, _ in parsed_models]
        n_repr = max(ns) if ns else 0
        cells = []
        for nm, _, _ in parsed_models:
            acc = summary[nm]['pattern_acc'].get(pat, None)
            cells.append(f'{acc*100:>{width-1}.2f}%' if acc is not None else f'{"n/a":>{width}}')
        print(f'{pat:<14}  {n_repr:>8}  ' + '  '.join(cells))

    # Per-level table
    levels = sorted({
        lv for nm, _, _ in parsed_models for lv in summary[nm]['per_level']
    }, key=lambda s: int(s) if s.isdigit() else 0)
    if levels:
        print('\n=== Per-level accuracy ===\n')
        print(f'{"level":<8}  ' + '  '.join(f'{n:>{width}}' for n, _, _ in parsed_models))
        for lv in levels:
            cells = []
            for nm, _, _ in parsed_models:
                acc = summary[nm]['per_level'].get(lv)
                cells.append(f'{acc*100:>{width-1}.2f}%' if acc is not None else f'{"n/a":>{width}}')
            print(f'{lv:<8}  ' + '  '.join(cells))

    # Confusion summary
    print('\n=== Confusion matrices (rows=truth L/R/E, cols=pred) ===\n')
    for nm, _, _ in parsed_models:
        print(f'-- {nm} --')
        for i, label in enumerate(['L', 'R', 'E']):
            print(f'  {label}: ' + ' '.join(f'{x:>6}' for x in summary[nm]['confusion'][i]))

    out = {
        'baseline': a.baseline,
        'n_charts': len(baseline),
        'models': {nm: {'dir': mdir, 'backend_override': bo}
                   for nm, mdir, bo in parsed_models},
        'summary': summary,
        'per_chart': per_model_results,
    }
    if a.output_json:
        os.makedirs(os.path.dirname(a.output_json) or '.', exist_ok=True)
        with open(a.output_json, 'w') as f:
            json.dump(out, f, indent=2)
        logger.success(f'JSON written: {a.output_json}')

    if a.plot:
        os.makedirs(os.path.dirname(a.plot) or '.', exist_ok=True)
        make_confusion_plot(confusion_matrices, a.plot)


if __name__ == '__main__':
    main()
