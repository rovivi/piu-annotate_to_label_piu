from __future__ import annotations
"""
benchmark_annotations.py

Compare processed_db annotations against vis-ss reference for all matching charts.
Reports overall accuracy, triple accuracy, jack/repeated-step accuracy.
Also simulates the naturalness-based multi-note fix to show potential improvement.

Usage:
    python scripts/benchmark_annotations.py
    python scripts/benchmark_annotations.py --vis_dir /path/to/vis-ss/chart-jsons/120524
    python scripts/benchmark_annotations.py --song Mad5cience  # filter by song name
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict


VIS_DIR = '/Users/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524'
PROC_DIR = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/processed_db'

try:
    from piu_annotate.formats import notelines as _notelines
    _NOTELINES_AVAILABLE = True
except ImportError:
    _NOTELINES_AVAILABLE = False


def _natural_score(arrow_positions: list[int], combo: tuple[int], is_singles: bool) -> int:
    mid = 2.5 if is_singles else 4.5
    return sum(1 for p, l in zip(arrow_positions, combo) if (p < mid and l == 0) or (p > mid and l == 1))


def simulate_fix(taps: list, is_singles: bool) -> list:
    """Apply naturalness-based multi-note fix (mirrors process_db_matches._fix_multihits_by_naturalness)."""
    if not _NOTELINES_AVAILABLE:
        return taps
    L2I = {'l': 0, 'r': 1, 'e': 0, 'h': 1, '?': 0}
    I2L = {0: 'l', 1: 'r'}
    t2i: dict[float, list[int]] = defaultdict(list)
    for i, tap in enumerate(taps):
        t2i[round(tap[1], 5)].append(i)
    taps = [list(t) for t in taps]
    for t, idxs in t2i.items():
        if len(idxs) < 2:
            continue
        panels = [taps[i][0] for i in idxs]
        limbs = [L2I.get(taps[i][2], 0) for i in idxs]
        order = sorted(range(len(panels)), key=lambda x: panels[x])
        ap = [panels[o] for o in order]
        sl = [limbs[o] for o in order]
        si = [idxs[o] for o in order]
        vc = _notelines.multihit_to_valid_feet(ap)
        if not vc:
            continue
        cur = tuple(sl)
        best = max(vc, key=lambda c: (_natural_score(ap, c, is_singles), -sum(a != b for a, b in zip(c, cur))))
        for idx, limb in zip(si, best):
            taps[idx][2] = I2L[limb]
    return taps


def load_vis_shortname_index(vis_dir: str) -> dict[str, str]:
    """Map shortname → vis-ss json path."""
    idx = {}
    for fn in os.listdir(vis_dir):
        if not fn.endswith('.json'):
            continue
        path = os.path.join(vis_dir, fn)
        try:
            d = json.load(open(path))
            sn = d[2].get('shortname')
            if sn:
                idx[sn] = path
        except Exception:
            pass
    return idx


def is_triple_tap(taps: list, t: float, threshold: float = 1e-3) -> bool:
    """True if time t has 3+ simultaneous taps."""
    return sum(1 for tap in taps if abs(tap[1] - t) < threshold) >= 3


def get_times_with_note_count(notes: list, threshold: float = 1e-3) -> dict:
    """Map time → count of notes at that time."""
    from collections import Counter
    times = [round(n[1], 6) for n in notes]
    return Counter(times)


def is_repeated_tap(taps: list, idx: int, threshold: float = 1e-3) -> bool:
    """True if this tap is on the same panel as the previous tap at a different time."""
    if idx == 0:
        return False
    curr_panel, curr_t = taps[idx][0], taps[idx][1]
    # find previous tap at a different time on same panel
    for j in range(idx - 1, -1, -1):
        prev_panel, prev_t = taps[j][0], taps[j][1]
        if abs(prev_t - curr_t) < threshold:
            continue  # same-time multi (skip)
        # different time: check if same panel
        return prev_panel == curr_panel
    return False


def compare_chart(proc_cjs: list, ref: list, is_singles: bool = True) -> dict:
    """Compare proc_cjs taps/holds against reference. Returns accuracy stats.
    Also includes stats for the simulated naturalness fix applied to proc_taps.
    """
    ref_taps, ref_holds = ref[0], ref[1]
    proc_taps, proc_holds = proc_cjs[0], proc_cjs[1]

    stats = defaultdict(int)

    if len(ref_taps) != len(proc_taps):
        stats['structure_mismatch'] = 1
        return stats

    # count notes per time for triple detection
    time_to_count = get_times_with_note_count(ref_taps)

    # Simulate fix on proc_taps
    fixed_taps = simulate_fix(proc_taps, is_singles)

    # TAP comparison
    for i, (ref_tap, proc_tap, fix_tap) in enumerate(zip(ref_taps, proc_taps, fixed_taps)):
        if ref_tap[0] != proc_tap[0] or abs(ref_tap[1] - proc_tap[1]) > 1e-3:
            stats['tap_content_mismatch'] += 1
            continue

        t = round(ref_tap[1], 6)
        count_at_t = time_to_count[t]
        is_triple = count_at_t >= 3
        is_repeated = is_repeated_tap(ref_taps, i)

        correct = int(ref_tap[2] == proc_tap[2])
        fixed_correct = int(ref_tap[2] == fix_tap[2])

        stats['tap_total'] += 1
        stats['tap_correct'] += correct
        stats['tap_fixed_correct'] += fixed_correct

        if is_triple:
            stats['triple_tap_total'] += 1
            stats['triple_tap_correct'] += correct
            stats['triple_tap_fixed_correct'] += fixed_correct

        if is_repeated:
            stats['repeated_tap_total'] += 1
            stats['repeated_tap_correct'] += correct
            stats['repeated_tap_fixed_correct'] += fixed_correct

        if not is_triple and not is_repeated:
            stats['single_non_repeated_total'] += 1
            stats['single_non_repeated_correct'] += correct
            stats['single_non_repeated_fixed_correct'] += fixed_correct

    # HOLD comparison
    for ref_hold, proc_hold in zip(ref_holds, proc_holds):
        if ref_hold[0] != proc_hold[0]:
            stats['hold_content_mismatch'] += 1
            continue
        correct = int(ref_hold[3] == proc_hold[3])
        stats['hold_total'] += 1
        stats['hold_correct'] += correct

    return stats


def pct(num, denom) -> str:
    if denom == 0:
        return 'n/a'
    return f'{100.0 * num / denom:.1f}%'


def build_per_chart(proc_dir: str, vis_idx: dict, song_filter: str | None, mode_filter: str | None) -> tuple[list, defaultdict]:
    """Load all charts, compare against vis-ss, return (per_chart list, totals dict)."""
    proc_files = [f for f in os.listdir(proc_dir) if f.endswith('.json')]
    totals: defaultdict = defaultdict(int)
    per_chart = []

    for fname in proc_files:
        path = os.path.join(proc_dir, fname)
        try:
            proc = json.load(open(path))
        except Exception:
            continue

        proc_cjs = proc.get('cjs')
        if not proc_cjs or len(proc_cjs) < 3:
            continue

        meta = proc_cjs[2]
        shortname = meta.get('shortname', '')
        mode = proc.get('mode', '')
        song_name = proc.get('song_name', '')

        if song_filter and song_filter.lower() not in song_name.lower():
            continue
        if mode_filter and mode != mode_filter:
            continue

        ref_path = vis_idx.get(shortname)
        if not ref_path:
            continue

        try:
            ref = json.load(open(ref_path))
        except Exception:
            continue

        is_singles = (proc.get('mode', 'S') == 'S')
        stats = compare_chart(proc_cjs, ref, is_singles)
        if stats.get('structure_mismatch'):
            continue

        tap_acc = stats['tap_correct'] / max(stats['tap_total'], 1)
        level = proc.get('level')
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 0

        per_chart.append({
            'shortname': shortname,
            'song_name': song_name,
            'mode': mode,
            'level': level,
            'tap_acc': tap_acc,
            'stats': stats,
        })
        for k, v in stats.items():
            totals[k] += v

    return per_chart, totals


def plot_results(per_chart: list, totals: defaultdict, out_path: str = 'benchmark_charts.png'):
    """Generate a 2x2 matplotlib figure summarising benchmark results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PIU Limb Annotation – Benchmark Results', fontsize=14, fontweight='bold')

    # ── 1. Accuracy breakdown bar chart ──────────────────────────────────────
    ax = axes[0, 0]
    categories = ['Overall tap', 'Triple taps', 'Jacks/repeated', 'Single non-rep', 'Hold']
    keys_correct = ['tap_correct', 'triple_tap_correct', 'repeated_tap_correct',
                    'single_non_repeated_correct', 'hold_correct']
    keys_total   = ['tap_total',   'triple_tap_total',   'repeated_tap_total',
                    'single_non_repeated_total', 'hold_total']
    keys_fixed   = ['tap_fixed_correct', 'triple_tap_fixed_correct', 'repeated_tap_fixed_correct',
                    'single_non_repeated_fixed_correct', None]

    vals_cur, vals_fix = [], []
    for kc, kt, kf in zip(keys_correct, keys_total, keys_fixed):
        cur  = 100.0 * totals[kc]  / max(totals[kt], 1)
        fix  = 100.0 * totals[kf]  / max(totals[kt], 1) if kf else cur
        vals_cur.append(cur)
        vals_fix.append(fix)

    x = range(len(categories))
    bars1 = ax.bar([i - 0.2 for i in x], vals_cur, width=0.35, label='Current', color='steelblue')
    bars2 = ax.bar([i + 0.2 for i in x], vals_fix, width=0.35, label='With naturalness fix', color='darkorange', alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.5)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7)
    ax.legend(fontsize=8)
    ax.set_title('Accuracy by note type')

    # ── 2. Accuracy by difficulty level (scatter) ────────────────────────────
    ax = axes[0, 1]
    levels_s = [c['level'] for c in per_chart if c['mode'] == 'S']
    accs_s   = [c['tap_acc'] * 100 for c in per_chart if c['mode'] == 'S']
    levels_d = [c['level'] for c in per_chart if c['mode'] == 'D']
    accs_d   = [c['tap_acc'] * 100 for c in per_chart if c['mode'] == 'D']
    ax.scatter(levels_s, accs_s, alpha=0.3, s=12, color='steelblue', label='Singles')
    ax.scatter(levels_d, accs_d, alpha=0.3, s=12, color='firebrick', label='Doubles')

    # per-level median line
    from collections import defaultdict as dd
    lv_acc: dict = dd(list)
    for c in per_chart:
        lv_acc[c['level']].append(c['tap_acc'] * 100)
    sorted_lvs = sorted(lv_acc)
    medians = [float(sum(lv_acc[lv]) / len(lv_acc[lv])) for lv in sorted_lvs]
    ax.plot(sorted_lvs, medians, 'k-o', markersize=3, linewidth=1.2, label='Mean per level')

    ax.set_xlabel('Difficulty level')
    ax.set_ylabel('Tap accuracy (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_title('Tap accuracy vs difficulty level')

    # ── 3. Worst 30 charts horizontal bar ────────────────────────────────────
    ax = axes[1, 0]
    worst30 = sorted(per_chart, key=lambda c: c['tap_acc'])[:30]
    labels  = [f"{c['shortname'][:35]}  (lv{c['level']})" for c in worst30]
    vals    = [c['tap_acc'] * 100 for c in worst30]
    colors  = ['#d73027' if v < 50 else '#fc8d59' if v < 70 else '#fee090' for v in vals]
    bars = ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('Tap accuracy (%)')
    ax.set_xlim(0, 105)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title('30 worst charts by tap accuracy')
    patches = [mpatches.Patch(color='#d73027', label='<50%'),
               mpatches.Patch(color='#fc8d59', label='50–70%'),
               mpatches.Patch(color='#fee090', label='70–80%')]
    ax.legend(handles=patches, fontsize=7, loc='lower right')

    # ── 4. Top 30 hardest charts accuracy ────────────────────────────────────
    ax = axes[1, 1]
    hardest30 = sorted(per_chart, key=lambda c: (-c['level'], c['tap_acc']))[:30]
    labels_h  = [f"{c['shortname'][:35]}  (lv{c['level']})" for c in hardest30]
    vals_h    = [c['tap_acc'] * 100 for c in hardest30]
    jacks_h   = [100.0 * c['stats']['repeated_tap_correct'] / max(c['stats']['repeated_tap_total'], 1)
                 for c in hardest30]
    ax.barh(range(len(labels_h)), vals_h, color='steelblue', alpha=0.7, label='Overall tap')
    ax.barh(range(len(labels_h)), jacks_h, color='firebrick', alpha=0.6, label='Jack/repeated')
    ax.set_yticks(range(len(labels_h)))
    ax.set_yticklabels(labels_h, fontsize=6)
    ax.set_xlabel('Accuracy (%)')
    ax.set_xlim(0, 105)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.5)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Top 30 hardest charts: overall vs jack accuracy')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nChart saved to: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_dir', default=VIS_DIR)
    parser.add_argument('--proc_dir', default=PROC_DIR)
    parser.add_argument('--song', default=None, help='Filter by song name substring')
    parser.add_argument('--mode', default=None, help='Filter by mode: S or D')
    parser.add_argument('--show_worst', type=int, default=10, help='Show N worst charts by tap accuracy')
    parser.add_argument('--top_diff', type=int, default=30, help='Show top N hardest charts')
    parser.add_argument('--plot', action='store_true', help='Generate matplotlib charts (saved to benchmark_charts.png)')
    parser.add_argument('--plot_out', default='benchmark_charts.png', help='Output path for plot')
    parser.add_argument('--diff_viewer', action='store_true',
                        help='Regenerar diff viewer HTML tras el benchmark y abrirlo')
    parser.add_argument('--n_viewer_charts', type=int, default=40,
                        help='Cuántos charts incluir en el diff viewer por modo (default 40)')
    args = parser.parse_args()

    print(f'Loading vis-ss index from {args.vis_dir}...')
    vis_idx = load_vis_shortname_index(args.vis_dir)
    print(f'  {len(vis_idx)} vis-ss charts indexed')

    proc_files = [f for f in os.listdir(args.proc_dir) if f.endswith('.json')]
    print(f'  {len(proc_files)} processed_db charts')

    per_chart, totals = build_per_chart(args.proc_dir, vis_idx, args.song, args.mode)

    print(f'\n=== BENCHMARK: {len(per_chart)} matched charts ===\n')
    print(f'  {"":35}  {"current":>7}  {"with_fix":>8}')
    print(f'  {"-"*55}')

    def row(label, correct, total, fixed_correct=None, fixed_total=None):
        ft = fixed_total if fixed_total is not None else total
        fix_str = f'  {pct(fixed_correct, ft):>8}' if fixed_correct is not None else ''
        print(f'  {label:<35} {pct(correct, total):>7}{fix_str}  ({correct}/{total})')

    row('Tap accuracy', totals['tap_correct'], totals['tap_total'],
        totals['tap_fixed_correct'], totals['tap_total'])
    row('  - Triple taps', totals['triple_tap_correct'], totals['triple_tap_total'],
        totals['triple_tap_fixed_correct'], totals['triple_tap_total'])
    row('  - Repeated/jack taps', totals['repeated_tap_correct'], totals['repeated_tap_total'],
        totals['repeated_tap_fixed_correct'], totals['repeated_tap_total'])
    row('  - Single non-repeated', totals['single_non_repeated_correct'], totals['single_non_repeated_total'],
        totals['single_non_repeated_fixed_correct'], totals['single_non_repeated_total'])
    row('Hold accuracy', totals['hold_correct'], totals['hold_total'])

    print(f'\n=== WORST {args.show_worst} CHARTS by tap accuracy ===\n')
    worst = sorted(per_chart, key=lambda x: x['tap_acc'])[:args.show_worst]
    for c in worst:
        s = c['stats']
        triple_acc = pct(s['triple_tap_correct'], s['triple_tap_total'])
        jack_acc = pct(s['repeated_tap_correct'], s['repeated_tap_total'])
        print(f"  {pct(s['tap_correct'], s['tap_total']):>7}  "
              f"triple={triple_acc}  jack={jack_acc}  "
              f"{c['shortname']}")

    print(f'\n=== TOP {args.top_diff} HARDEST CHARTS (by level) ===\n')
    hardest = sorted(per_chart, key=lambda x: (-x['level'], x['tap_acc']))[:args.top_diff]
    print(f'  {"shortname":<45}  {"lv":>3}  {"tap":>7}  {"jack":>7}  {"triple":>7}')
    print(f'  {"-"*80}')
    for c in hardest:
        s = c['stats']
        print(f"  {c['shortname']:<45}  {c['level']:>3}  "
              f"{pct(s['tap_correct'], s['tap_total']):>7}  "
              f"{pct(s['repeated_tap_correct'], s['repeated_tap_total']):>7}  "
              f"{pct(s['triple_tap_correct'], s['triple_tap_total']):>7}")

    # Summary by mode
    print('\n=== BY MODE ===\n')
    for m in ['S', 'D']:
        charts_m = [c for c in per_chart if c['mode'] == m]
        if not charts_m:
            continue
        tap_c = sum(c['stats']['tap_correct'] for c in charts_m)
        tap_t = sum(c['stats']['tap_total'] for c in charts_m)
        tri_c = sum(c['stats']['triple_tap_correct'] for c in charts_m)
        tri_t = sum(c['stats']['triple_tap_total'] for c in charts_m)
        rep_c = sum(c['stats']['repeated_tap_correct'] for c in charts_m)
        rep_t = sum(c['stats']['repeated_tap_total'] for c in charts_m)
        print(f'  Mode {m} ({len(charts_m)} charts):')
        row('  Tap overall', tap_c, tap_t)
        row('  Triple taps', tri_c, tri_t)
        row('  Repeated/jack taps', rep_c, rep_t)

    if args.plot:
        plot_results(per_chart, totals, args.plot_out)

    if args.diff_viewer:
        import subprocess
        script = os.path.join(os.path.dirname(__file__), 'generate_diff_viewer.py')
        cmd = [
            sys.executable, script,
            '--n_charts', str(args.n_viewer_charts),
            '--mode', 'all',
        ]
        if args.song:
            cmd += ['--song', args.song]
        print(f'\nRegenerando diff viewer...')
        subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
