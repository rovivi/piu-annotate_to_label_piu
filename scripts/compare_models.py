from __future__ import annotations
"""
compare_models.py

Re-runs limb inference with the NEW models on the 54 baseline charts
(30 worst accuracy + 30 hardest level) and compares against the OLD
processed_db results and vis-ss ground truth.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --baseline artifacts/benchmark_baseline_60.json
    python scripts/compare_models.py --plot
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from collections import defaultdict
from hackerargs import args as hargs
from loguru import logger

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml import predictor as ml_predictor

VIS_DIR   = '/Users/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524'
PROC_DIR  = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/processed_db'
CSV_DIR   = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524'
MODEL_DIR = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss'
BASELINE  = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/benchmark_baseline_60.json'


def setup_model_args(sd: str):
    hargs['model'] = 'lightgbm'
    hargs['model.dir'] = MODEL_DIR
    hargs[f'model.arrows_to_limb-{sd}']    = f'{sd}-arrows_to_limb.txt'
    hargs[f'model.arrowlimbs_to_limb-{sd}'] = f'{sd}-arrowlimbs_to_limb.txt'
    hargs[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-arrows_to_matchnext.txt'
    hargs[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-arrows_to_matchprev.txt'


def load_vis_index(vis_dir: str) -> dict[str, str]:
    idx = {}
    for fn in os.listdir(vis_dir):
        if not fn.endswith('.json'):
            continue
        try:
            d = json.load(open(os.path.join(vis_dir, fn)))
            sn = d[2].get('shortname')
            if sn:
                idx[sn] = os.path.join(vis_dir, fn)
        except Exception:
            pass
    return idx


def find_csv_for_shortname(shortname: str) -> str | None:
    fname = shortname + '.csv'
    path = os.path.join(CSV_DIR, fname)
    return path if os.path.isfile(path) else None


def score_limbs_vs_ref(pred_limbs_binary, ref_taps, pred_coords) -> dict:
    """Compare predicted limbs against vis-ss reference taps."""
    L2I = {'l': 0, 'r': 1, 'e': 0, 'h': 1, '?': -1}
    stats = defaultdict(int)

    if len(pred_limbs_binary) != len(pred_coords):
        return stats

    ref_by_time: dict[float, list] = defaultdict(list)
    for tap in ref_taps:
        ref_by_time[round(tap[1], 5)].append(tap)

    time_count = defaultdict(int)
    for tap in ref_taps:
        time_count[round(tap[1], 6)] += 1

    prev_panel_to_time: dict[int, float] = {}

    for pc_idx, pc in enumerate(pred_coords):
        row = None
        t_key = None
        for tap in ref_taps:
            if tap[0] == pc.arrow_pos:
                t_key = round(tap[1], 5)
                row = tap
                break

        if row is None:
            continue

        ref_limb_int = L2I.get(row[2], -1)
        if ref_limb_int < 0:
            continue

        pred_limb_int = int(pred_limbs_binary[pc_idx])
        correct = int(pred_limb_int == ref_limb_int)

        t6 = round(row[1], 6)
        is_triple = time_count[t6] >= 3
        panel = pc.arrow_pos
        prev_t = prev_panel_to_time.get(panel)
        is_jack = prev_t is not None and abs(prev_t - row[1]) > 1e-3
        prev_panel_to_time[panel] = row[1]

        stats['tap_total'] += 1
        stats['tap_correct'] += correct
        if is_triple:
            stats['triple_total'] += 1
            stats['triple_correct'] += correct
        if is_jack:
            stats['jack_total'] += 1
            stats['jack_correct'] += correct

    return stats


def pct(n, d) -> str:
    return 'n/a' if d == 0 else f'{100.0 * n / d:.1f}%'


def delta(new_val: float | None, old_val: float) -> str:
    if new_val is None:
        return '  n/a'
    diff = new_val - old_val
    sign = '+' if diff >= 0 else ''
    return f'{sign}{diff:.1f}pp'


def run_inference_on_csv(csv_path: str, sd: str) -> tuple | None:
    """Run new-model inference on a single chartstructs CSV.
    Returns (pred_limbs, pred_coords) or None on failure.
    """
    try:
        cs = ChartStruct.from_file(csv_path)
        model_suite = ModelSuite(sd)
        cs_out = ml_predictor.predict(cs, model_suite)
        fcs_out = cs_out
        pred_coords = cs.get_prediction_coordinates()
        limb_col = cs_out.df.get('Limb annotation', None)
        if limb_col is None:
            return None
        L2I = {'l': 0, 'r': 1, 'e': 0, 'h': 1, '?': 0}
        pred_limbs = []
        for pc in pred_coords:
            row = cs_out.df.iloc[pc.row_idx]
            val = row['Limb annotation']
            if isinstance(val, str):
                limb_char = val[pc.limb_idx] if pc.limb_idx < len(val) else 'l'
            else:
                limb_char = 'l'
            pred_limbs.append(L2I.get(limb_char, 0))
        return np.array(pred_limbs), pred_coords
    except Exception as e:
        logger.warning(f'Inference failed for {csv_path}: {e}')
        return None


def make_plot(results: list, out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    worst_acc = [r for r in results if 'worst_accuracy' in r['groups'] and r.get('new_tap') is not None]
    hardest   = [r for r in results if 'hardest_level'  in r['groups'] and r.get('new_tap') is not None]

    def sort_and_plot(ax, items, sort_key, title, color_old, color_new):
        items = sorted(items, key=sort_key)
        labels = [f"{r['shortname'][:32]}  (lv{r['level']})" for r in items]
        old_vals = [r['old_tap'] for r in items]
        new_vals = [r['new_tap'] for r in items]
        y = range(len(labels))
        ax.barh([i + 0.2 for i in y], new_vals, height=0.35, color=color_new, alpha=0.85, label='New model')
        ax.barh([i - 0.2 for i in y], old_vals, height=0.35, color=color_old, alpha=0.6,  label='Old model')
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel('Tap accuracy (%)')
        ax.set_xlim(0, 105)
        ax.axvline(100, color='gray', linestyle='--', linewidth=0.5)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.set_title(title)

    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    fig.suptitle('Model comparison: Old vs New — Tap accuracy', fontsize=13, fontweight='bold')
    sort_and_plot(axes[0], worst_acc, lambda r: r['old_tap'],
                  '30 worst accuracy charts', '#d73027', '#1a9850')
    sort_and_plot(axes[1], hardest,   lambda r: (-r['level'], r['old_tap']),
                  '30 hardest charts (by level)', '#4575b4', '#74add1')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nChart saved to: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_out', default='artifacts/model_comparison.png')
    pargs = parser.parse_args()

    baseline = json.load(open(pargs.baseline))
    vis_idx = load_vis_index(VIS_DIR)

    # Pre-load model suites once each
    setup_model_args('singles')
    suite_s = ModelSuite('singles')
    setup_model_args('doubles')
    suite_d = ModelSuite('doubles')

    results = []
    for entry in baseline:
        sn = entry['shortname']
        sd = 'singles' if entry['mode'] == 'S' else 'doubles'

        csv_path = find_csv_for_shortname(sn)
        ref_path = vis_idx.get(sn)

        result = {
            'shortname': sn,
            'song_name': entry['song_name'],
            'mode': entry['mode'],
            'level': entry['level'],
            'groups': entry['groups'],
            'old_tap': entry['tap_acc_old'],
            'old_jack': entry['jack_acc_old'],
            'old_triple': entry['triple_acc_old'],
            'new_tap': None,
            'new_jack': None,
            'new_triple': None,
        }

        if not csv_path:
            logger.warning(f'No CSV found for {sn}')
            results.append(result)
            continue
        if not ref_path:
            logger.warning(f'No vis-ss ref for {sn}')
            results.append(result)
            continue

        try:
            ref = json.load(open(ref_path))
            ref_taps = ref[0]
        except Exception as e:
            logger.warning(f'Could not load ref for {sn}: {e}')
            results.append(result)
            continue

        suite = suite_s if sd == 'singles' else suite_d

        # need fresh ChartStruct per inference call
        try:
            cs = ChartStruct.from_file(csv_path)
            from piu_annotate.ml import featurizers as ftz
            _cs, fcs, pred_limbs = ml_predictor.predict(cs, suite)

            # Overall accuracy (vs 'Limb annotation' col in CSV = vis-ss ground truth)
            eval_dict = fcs.evaluate(pred_limbs)
            result['new_tap'] = round(eval_dict['accuracy-float'] * 100, 2)

            # Jack and triple accuracy
            labels = fcs.get_labels_from_limb_col('Limb annotation')
            pred_coords = fcs.pred_coords

            time_count: dict[float, int] = defaultdict(int)
            for pc in pred_coords:
                t = round(float(cs.df.at[pc.row_idx, 'Time']), 5)
                time_count[t] += 1

            panel_to_last_t: dict[int, float] = {}
            jack_correct = jack_total = 0
            triple_correct = triple_total = 0

            for idx, pc in enumerate(pred_coords):
                t = round(float(cs.df.at[pc.row_idx, 'Time']), 5)
                correct = int(pred_limbs[idx] == labels[idx])
                is_triple = time_count[t] >= 3
                prev_t = panel_to_last_t.get(pc.arrow_pos)
                is_jack = prev_t is not None and abs(prev_t - t) > 1e-3
                panel_to_last_t[pc.arrow_pos] = t
                if is_triple:
                    triple_total += 1
                    triple_correct += correct
                if is_jack:
                    jack_total += 1
                    jack_correct += correct

            result['new_jack']   = round(100.0 * jack_correct   / max(jack_total,   1), 2) if jack_total   else None
            result['new_triple'] = round(100.0 * triple_correct / max(triple_total, 1), 2) if triple_total else None
            logger.info(f"{sn[:45]}  old={result['old_tap']:.1f}%  new={result['new_tap']:.1f}%")
        except Exception as e:
            logger.warning(f'Failed inference on {sn}: {e}')

        results.append(result)

    # Print comparison tables
    for group, label in [('worst_accuracy', 'WORST 30 (low accuracy)'), ('hardest_level', 'HARDEST 30 (high level)')]:
        subset = [r for r in results if group in r['groups']]
        subset = sorted(subset, key=lambda r: r['old_tap'] if group == 'worst_accuracy' else (-r['level'], r['old_tap']))

        print(f'\n=== {label} ===\n')
        print(f'  {"shortname":<45}  {"lv":>3}  {"old tap":>7}  {"new tap":>7}  {"delta":>7}  {"old jack":>8}  {"new jack":>8}')
        print(f'  {"-"*100}')
        for r in subset:
            d = delta(r['new_tap'], r['old_tap']) if r['new_tap'] is not None else '   n/a'
            new_tap_s  = f"{r['new_tap']:.1f}%"  if r['new_tap']  is not None else '   n/a'
            new_jack_s = f"{r['new_jack']:.1f}%" if r['new_jack'] is not None else '   n/a'
            print(f"  {r['shortname']:<45}  {r['level']:>3}  {r['old_tap']:>6.1f}%  {new_tap_s:>7}  {d:>7}  {r['old_jack']:>7.1f}%  {new_jack_s:>8}")

    # Summary
    all_with_new = [r for r in results if r['new_tap'] is not None]
    if all_with_new:
        avg_old = sum(r['old_tap'] for r in all_with_new) / len(all_with_new)
        avg_new = sum(r['new_tap'] for r in all_with_new) / len(all_with_new)
        improved = sum(1 for r in all_with_new if r['new_tap'] > r['old_tap'])
        print(f'\n=== SUMMARY ({len(all_with_new)} charts) ===')
        print(f'  Avg tap accuracy:  old={avg_old:.1f}%  new={avg_new:.1f}%  delta={avg_new - avg_old:+.1f}pp')
        print(f'  Improved: {improved}/{len(all_with_new)} charts')

    if pargs.plot:
        make_plot(results, pargs.plot_out)


if __name__ == '__main__':
    main()
