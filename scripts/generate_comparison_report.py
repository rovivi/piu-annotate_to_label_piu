#!/usr/bin/env python3
"""Render a self-contained HTML report from compare_models.py JSON output.

Replaces ``generate_v8_comparison_report.py``, ``..._simple.py`` and
``generate_v9_ft_comparison_report.py`` with one script that works for any
number of models.

Usage
-----
    # 1. Run multi-model comparison
    python scripts/compare_models.py \\
        --models lgbm:artifacts/models/visss \\
        --models mlx_v8:artifacts/models/visss-mlx-v8 \\
        --output_json artifacts/comparator/data.json

    # 2. Render report
    python scripts/generate_comparison_report.py \\
        --data artifacts/comparator/data.json \\
        --out artifacts/comparator/report.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import defaultdict


HEADER = '''<!doctype html>
<html lang="es"><head>
<meta charset="utf-8">
<title>Model Comparison Report</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --blue: #58a6ff; --green: #3fb950; --orange: #f0883e;
    --red: #f85149; --gold: #ffd700; --purple: #a371f7;
  }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: -apple-system, "Segoe UI", sans-serif; font-size: 14px; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 1.4rem; margin-bottom: 6px; }
  h2 { font-size: 1.1rem; margin-top: 32px; margin-bottom: 12px;
        border-bottom: 1px solid var(--border); padding-bottom: 6px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px;
          font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); }
  th { text-align: left; background: var(--bg2); }
  td.acc.high   { color: var(--green); font-weight: 600; }
  td.acc.mid    { color: var(--gold); }
  td.acc.low    { color: var(--red); }
  td.acc.best   { background: rgba(63, 185, 80, 0.08); }
  td.num { text-align: right; }
  .meta { color: var(--muted); font-size: 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
  .conf-card { background: var(--bg2); border: 1px solid var(--border);
               border-radius: 6px; padding: 12px; }
  .conf-card h3 { margin: 0 0 8px 0; font-size: 14px; }
  .conf-table { width: 100%; }
  .conf-table td { text-align: center; padding: 6px; border: 1px solid var(--border); }
  .conf-table td.label { background: var(--bg3); font-weight: 600; color: var(--muted); }
  .bar { display: inline-block; height: 8px; background: var(--blue);
         vertical-align: middle; border-radius: 3px; min-width: 1px; }
  .row { display: flex; align-items: center; gap: 8px; }
  .row .label { flex: 0 0 120px; }
  .row .val { flex: 0 0 60px; text-align: right; font-variant-numeric: tabular-nums; }
</style>
</head><body><div class="container">
'''

FOOTER = '</div></body></html>'


def _acc_class(acc: float, best_acc: float | None) -> str:
    cls = []
    if acc >= 0.95:
        cls.append('high')
    elif acc >= 0.85:
        cls.append('mid')
    else:
        cls.append('low')
    if best_acc is not None and abs(acc - best_acc) < 1e-6:
        cls.append('best')
    return ' '.join(cls)


def render_pattern_table(summary: dict, models: list[str]) -> str:
    """Pattern accuracy table — one row per pattern bucket, one col per model."""
    pattern_keys = ['all', 'tap', 'jack', 'triple', 'jump',
                    'bracket_ll', 'bracket_rr', 'bracket_lr',
                    'hold_release', 'stream']
    rows = []
    for pat in pattern_keys:
        ns = [summary[m]['pattern_total'].get(pat, 0) for m in models]
        n_max = max(ns) if ns else 0
        accs = [summary[m]['pattern_acc'].get(pat) for m in models]
        best = max((a for a in accs if a is not None), default=None)
        cells = []
        for m, a in zip(models, accs):
            if a is None:
                cells.append('<td class="num">n/a</td>')
            else:
                cells.append(f'<td class="num acc {_acc_class(a, best)}">{a*100:.2f}%</td>')
        rows.append(f'<tr><td>{html.escape(pat)}</td><td class="num">{n_max}</td>' + ''.join(cells) + '</tr>')

    header_cells = ''.join(f'<th>{html.escape(m)}</th>' for m in models)
    return (
        '<table><thead><tr><th>pattern</th><th>n</th>'
        + header_cells
        + '</tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table>'
    )


def render_level_table(summary: dict, models: list[str]) -> str:
    levels = sorted(
        {lv for m in models for lv in summary[m]['per_level']},
        key=lambda s: int(s) if s.isdigit() else 0,
    )
    if not levels:
        return '<p class="meta">No per-level data.</p>'
    rows = []
    for lv in levels:
        accs = [summary[m]['per_level'].get(lv) for m in models]
        best = max((a for a in accs if a is not None), default=None)
        cells = []
        for a in accs:
            if a is None:
                cells.append('<td class="num">n/a</td>')
            else:
                cells.append(f'<td class="num acc {_acc_class(a, best)}">{a*100:.2f}%</td>')
        rows.append(f'<tr><td>{html.escape(lv)}</td>' + ''.join(cells) + '</tr>')
    header_cells = ''.join(f'<th>{html.escape(m)}</th>' for m in models)
    return (
        '<table><thead><tr><th>level</th>'
        + header_cells
        + '</tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table>'
    )


def render_confusion_grid(summary: dict, models: list[str]) -> str:
    cards = []
    for m in models:
        conf = summary[m]['confusion']
        row_sums = [sum(r) for r in conf]
        cells = []
        for i, label in enumerate(['L', 'R', 'E']):
            cells.append(f'<tr><td class="label">{label}</td>')
            for j in range(3):
                v = conf[i][j]
                pct = (v / row_sums[i] * 100) if row_sums[i] else 0.0
                bg = f'rgba(63, 185, 80, {min(pct / 100, 1):.2f})' if i == j \
                    else f'rgba(248, 81, 73, {min(pct / 100, 1):.2f})'
                cells.append(
                    f'<td style="background: {bg}">'
                    f'{pct:.1f}%<br><span class="meta">{v}</span></td>'
                )
            cells.append('</tr>')
        cards.append(
            f'<div class="conf-card"><h3>{html.escape(m)}</h3>'
            f'<table class="conf-table"><thead><tr><th></th><th>L</th><th>R</th><th>E</th></tr></thead>'
            f'<tbody>{"".join(cells)}</tbody></table></div>'
        )
    return '<div class="grid">' + ''.join(cards) + '</div>'


def render_chart_table(per_chart: dict, models: list[str], top_n: int = 30) -> str:
    """Per-chart breakdown with delta vs first model. Sorted by worst delta."""
    if not per_chart:
        return ''
    first = models[0]
    chart_index: dict[str, dict] = {}
    for m in models:
        for row in per_chart.get(m, []):
            sn = row['shortname']
            if sn not in chart_index:
                chart_index[sn] = {
                    'shortname': sn,
                    'level': row.get('level'),
                    'mode': row.get('mode'),
                    'song_name': row.get('song_name'),
                }
            n_all = row['pattern_total'].get('all', 0)
            c_all = row['pattern_correct'].get('all', 0)
            chart_index[sn][m] = c_all / max(n_all, 1)

    rows = list(chart_index.values())
    rows = [r for r in rows if first in r]
    rows.sort(key=lambda r: r.get(first, 1.0))
    rows = rows[:top_n]

    header = ''.join(f'<th>{html.escape(m)}</th>' for m in models)
    out_rows = []
    for r in rows:
        acc_cells = []
        accs_for_row = [r.get(m) for m in models]
        best = max((a for a in accs_for_row if a is not None), default=None)
        for a in accs_for_row:
            if a is None:
                acc_cells.append('<td class="num">n/a</td>')
            else:
                acc_cells.append(f'<td class="num acc {_acc_class(a, best)}">{a*100:.1f}%</td>')
        out_rows.append(
            f'<tr><td>{html.escape(r["shortname"][:60])}</td>'
            f'<td class="num">{r.get("level", "?")}</td>'
            f'<td>{html.escape(r.get("mode", "?"))}</td>'
            + ''.join(acc_cells) + '</tr>'
        )

    return (
        '<table><thead><tr><th>chart</th><th>level</th><th>mode</th>'
        + header
        + '</tr></thead><tbody>'
        + ''.join(out_rows)
        + '</tbody></table>'
    )


def render(data: dict) -> str:
    summary = data['summary']
    per_chart = data.get('per_chart', {})
    models = list(summary.keys())

    parts = [HEADER]
    parts.append(
        f'<h1>Model Comparison Report</h1>'
        f'<div class="meta">Baseline: {html.escape(data.get("baseline", "?"))}'
        f' · {data.get("n_charts", "?")} charts'
        f' · {len(models)} models</div>'
    )
    parts.append('<h2>Pattern accuracy</h2>')
    parts.append(render_pattern_table(summary, models))

    parts.append('<h2>Per-level accuracy</h2>')
    parts.append(render_level_table(summary, models))

    parts.append('<h2>Confusion matrices (rows = truth, cols = pred)</h2>')
    parts.append(render_confusion_grid(summary, models))

    parts.append(f'<h2>Worst {30} charts (sorted by {models[0]} accuracy)</h2>')
    parts.append(render_chart_table(per_chart, models, top_n=30))

    parts.append(FOOTER)
    return ''.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='JSON output of compare_models.py')
    p.add_argument('--out', required=True, help='HTML report output path')
    a = p.parse_args()

    with open(a.data) as f:
        data = json.load(f)
    html_str = render(data)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        f.write(html_str)
    print(f'Report written: {a.out}')


if __name__ == '__main__':
    main()
