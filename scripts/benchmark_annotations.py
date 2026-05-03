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


VIS_DIR = '/home/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524'
PROC_DIR = '/home/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/processed_db'

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_dir', default=VIS_DIR)
    parser.add_argument('--proc_dir', default=PROC_DIR)
    parser.add_argument('--song', default=None, help='Filter by song name substring')
    parser.add_argument('--mode', default=None, help='Filter by mode: S or D')
    parser.add_argument('--show_worst', type=int, default=10, help='Show N worst charts by tap accuracy')
    args = parser.parse_args()

    print(f'Loading vis-ss index from {args.vis_dir}...')
    vis_idx = load_vis_shortname_index(args.vis_dir)
    print(f'  {len(vis_idx)} vis-ss charts indexed')

    proc_files = [f for f in os.listdir(args.proc_dir) if f.endswith('.json')]
    print(f'  {len(proc_files)} processed_db charts')

    totals = defaultdict(int)
    per_chart = []

    for fname in proc_files:
        path = os.path.join(args.proc_dir, fname)
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

        if args.song and args.song.lower() not in song_name.lower():
            continue
        if args.mode and mode != args.mode:
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
        per_chart.append({
            'shortname': shortname,
            'song_name': song_name,
            'mode': mode,
            'level': proc.get('level', '?'),
            'tap_acc': tap_acc,
            'stats': stats,
        })

        for k, v in stats.items():
            totals[k] += v

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


if __name__ == '__main__':
    main()
