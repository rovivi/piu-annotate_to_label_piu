#!/usr/bin/env python3
"""
generate_v8_comparison_report.py

Generates an HTML comparison report for PIU v8 limb annotation model.
Compares v8 predictions against vis-ss ground truth on benchmark charts.

Usage:
    python3 scripts/generate_v8_comparison_report.py
    python3 scripts/generate_v8_comparison_report.py --limit 10
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import functools
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mlx.core as mx
from hackerargs import args as hargs
from loguru import logger

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml import predictor as ml_predictor
from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer

VIS_DIR   = '/Users/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524'
CSV_DIR   = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524'
MODEL_DIR = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss-mlx-v8'
BASELINE  = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/benchmark_baseline_60.json'
OUTPUT   = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/v8_model_comparison_report.html'

# v8 model architecture (from train_config.json)
V8_CONFIG = {
    'd_model': 384,
    'n_heads': 8,
    'n_layers': 8,
    'ffn_dim': 1536,
    'input_dim': 22,  # 18 arrow features + 4 prev_limb one-hot
}


def create_v8_model(model_dir: str):
    """Create and load the v8 model with correct architecture."""
    model_file = os.path.join(model_dir, 'singles-arrows_to_limb-mlx-best.safetensors')
    model = LimbSequenceTransformer(
        input_dim=V8_CONFIG['input_dim'],
        d_model=V8_CONFIG['d_model'],
        n_heads=V8_CONFIG['n_heads'],
        n_layers=V8_CONFIG['n_layers'],
        ffn_dim=V8_CONFIG['ffn_dim'],
    )
    model.load_weights(model_file)
    return model


def predict_v8(model, points: np.ndarray) -> np.ndarray:
    """Run inference with v8 model."""
    N = len(points)
    chunk_size = 512
    overlap = 128
    stride = chunk_size - overlap

    @functools.lru_cache(maxsize=None)
    def get_compiled_forward():
        def forward(x, mask):
            return mx.softmax(model(x, padding_mask=mask), axis=-1)
        return mx.compile(forward)

    compiled_forward = get_compiled_forward()

    if N <= chunk_size:
        x_padded = np.zeros((chunk_size, points.shape[1]), dtype=np.float32)
        x_padded[:N] = points
        x_mx = mx.array(x_padded)[None]

        mask_padded = np.ones((1, chunk_size), dtype=bool)
        mask_padded[0, :N] = False
        mask_mx = mx.array(mask_padded)

        p = np.array(compiled_forward(x_mx, mask_mx))[0]
        p = p[:N]
        return np.argmax(p[:, :2], axis=-1).astype(int)

    p_total = np.zeros((N, 3), dtype=np.float32)
    weight_total = np.zeros((N, 1), dtype=np.float32)
    window = np.ones((chunk_size, 1), dtype=np.float32)
    window[:overlap, 0] = np.linspace(0, 1, overlap)
    window[-overlap:, 0] = np.linspace(1, 0, overlap)

    for start in range(0, N, stride):
        end = min(start + chunk_size, N)
        actual_len = end - start

        chunk_points = points[start:end]
        x_padded = np.zeros((chunk_size, points.shape[1]), dtype=np.float32)
        x_padded[:actual_len] = chunk_points
        x_mx = mx.array(x_padded)[None]

        mask_padded = np.ones((1, chunk_size), dtype=bool)
        mask_padded[0, :actual_len] = False
        mask_mx = mx.array(mask_padded)

        p = np.array(compiled_forward(x_mx, mask_mx))[0]
        p = p[:actual_len]

        if actual_len == 1:
            p = p[None, :]

        w = window[:actual_len].copy()
        if start == 0:
            w[:overlap] = 1.0
        if end == N:
            w[-overlap:] = 1.0

        p_total[start:end] += p * w
        weight_total[start:end] += w

        if end == N:
            break

    p_total = p_total / np.maximum(weight_total, 1e-6)
    return np.argmax(p_total[:, :2], axis=-1).astype(int)


def setup_v8_model_args():
    hargs['model'] = 'mlx'
    hargs['model.dir'] = MODEL_DIR
    hargs['model.arrows_to_limb-singles']    = 'singles-arrows_to_limb-mlx-best.safetensors'
    hargs['model.arrowlimbs_to_limb-singles'] = 'singles-arrows_to_limb-mlx-best.safetensors'
    hargs['model.arrows_to_matchnext-singles'] = 'singles-arrows_to_limb-mlx-best.safetensors'
    hargs['model.arrows_to_matchprev-singles'] = 'singles-arrows_to_limb-mlx-best.safetensors'


def load_vis_index(vis_dir: str) -> dict[str, str]:
    idx = {}
    for fn in os.listdir(vis_dir):
        if not fn.endswith('.json'):
            continue
        try:
            d = json.load(open(os.path.join(vis_dir, fn)))
            sn = d[2].get('shortname') if len(d) >= 3 else None
            if sn:
                idx[sn] = os.path.join(vis_dir, fn)
        except Exception:
            pass
    return idx


def find_csv_for_shortname(shortname: str) -> str | None:
    fname = shortname + '.csv'
    path = os.path.join(CSV_DIR, fname)
    return path if os.path.isfile(path) else None


def classify_note_type(pred_coords, cs):
    time_count = defaultdict(int)
    for pc in pred_coords:
        t = round(float(cs.df.at[pc.row_idx, 'Time']), 5)
        time_count[t] += 1

    panel_to_last_t = {}
    classified = []
    for pc in pred_coords:
        t = round(float(cs.df.at[pc.row_idx, 'Time']), 5)
        is_triple = time_count[t] >= 3
        prev_t = panel_to_last_t.get(pc.arrow_pos)
        is_jack = prev_t is not None and abs(prev_t - t) > 1e-3
        panel_to_last_t[pc.arrow_pos] = t

        if is_triple:
            note_type = 'triple'
        elif is_jack:
            note_type = 'jack'
        else:
            note_type = 'single'
        classified.append(note_type)
    return classified


def build_report(results, overall_v8_acc, overall_old_acc, timestamp):
    v8_wins = 0
    v8_tied = 0
    total_cmp = 0
    for r in results:
        if r['v8_tap'] is not None and r['old_tap'] is not None:
            total_cmp += 1
            if r['v8_tap'] > r['old_tap']:
                v8_wins += 1
            elif abs(r['v8_tap'] - r['old_tap']) < 0.1:
                v8_tied += 1

    valid_results = [r for r in results if r['v8_tap'] is not None]
    best = max(valid_results, key=lambda r: r['v8_tap']) if valid_results else {}
    worst = min(valid_results, key=lambda r: r['v8_tap']) if valid_results else {}
    avg_v8 = sum(r['v8_tap'] for r in valid_results) / len(valid_results) if valid_results else 0

    v8_jack = 0
    v8_triple = 0
    v8_single = 0
    jack_count = 0
    triple_count = 0
    single_count = 0
    for r in valid_results:
        if r['v8_jack'] is not None:
            v8_jack += r['v8_jack']
            jack_count += 1
        if r['v8_triple'] is not None:
            v8_triple += r['v8_triple']
            triple_count += 1
        if r['v8_single'] is not None:
            v8_single += r['v8_single']
            single_count += 1
    v8_jack /= max(jack_count, 1)
    v8_triple /= max(triple_count, 1)
    v8_single /= max(single_count, 1)

    results_json = json.dumps(results, ensure_ascii=False)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PIU v8 Model Comparison Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #21262d;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --accent3: #f0883e;
    --danger: #f85149;
    --gold: #ffd700;
    --purple: #a371f7;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    font-size: 15px;
    line-height: 1.6;
}}
.live-banner {{
    background: linear-gradient(90deg, #0a1a0a, #0d2b0d, #0a1a0a);
    border-bottom: 2px solid var(--accent2);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    position: sticky;
    top: 0;
    z-index: 100;
}}
.live-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent2);
    box-shadow: 0 0 8px 2px rgba(63,185,80,0.6);
}}
.live-banner span {{ font-weight: 700; color: var(--accent2); letter-spacing: 0.05em; }}
.live-banner small {{ color: var(--text-muted); margin-left: auto; font-size: 12px; }}
header {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 56px 24px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
header::before {{
    content: '';
    position: absolute;
    top: -80px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(88,166,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}}
.header-tag {{
    display: inline-block;
    background: rgba(63,185,80,0.15);
    border: 1px solid var(--accent2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent2);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 20px;
}}
header h1 {{
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 12px;
}}
header h1 .pct {{
    background: linear-gradient(135deg, #ffd700, #ff8c00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.header-stats {{
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
}}
.hstat {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 20px;
    min-width: 130px;
}}
.hstat .val {{
    font-size: 1.6rem;
    font-weight: 800;
    display: block;
}}
.hstat .lbl {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }}
.hstat.gold .val {{ color: var(--gold); }}
.hstat.green .val {{ color: var(--accent2); }}
.hstat.blue .val {{ color: var(--accent); }}
main {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px 80px; }}
section {{ margin-bottom: 60px; }}
h2 {{
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}}
h2 .icon {{ font-size: 1.3rem; }}
h2 .badge {{
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    background: rgba(88,166,255,0.15);
    color: var(--accent);
    border: 1px solid rgba(88,166,255,0.3);
    margin-left: 8px;
}}
.card {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}}
.card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
}}
.card.success {{ border-color: var(--accent2); background: rgba(63,185,80,0.05); }}
.card.danger {{ border-color: var(--danger); background: rgba(248,81,73,0.05); }}
.chart-wrap {{
    position: relative;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}}
.chart-title {{
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 20px;
}}
.model-progress {{ margin-bottom: 16px; }}
.model-progress .head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}}
.model-progress .name {{ font-weight: 600; }}
.model-progress .pct-val {{ font-weight: 700; font-size: 1.1rem; }}
.pbar-track {{
    height: 12px;
    background: var(--bg3);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}}
.pbar-fill {{
    height: 100%;
    border-radius: 6px;
    transition: width 1s ease;
    position: relative;
}}
.pbar-fill::after {{
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 40px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3));
}}
.pbar-old .pbar-fill {{ background: linear-gradient(90deg, #6b7280, #9ca3af); }}
.pbar-v8 .pbar-fill {{ background: linear-gradient(90deg, #ffd700, #ff8c00); animation: shimmer 2s infinite; }}
@keyframes shimmer {{
    0% {{ filter: brightness(1); }}
    50% {{ filter: brightness(1.2); }}
    100% {{ filter: brightness(1); }}
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th {{
    background: var(--bg3);
    padding: 10px 14px;
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}}
td {{
    padding: 9px 14px;
    border-bottom: 1px solid rgba(48,54,61,0.6);
    font-family: 'SF Mono', 'Fira Code', monospace;
    white-space: nowrap;
}}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
tr.best-row td {{ background: rgba(255,215,0,0.06); }}
tr.best-row td:first-child {{ border-left: 3px solid var(--gold); }}
tr.worst-row td {{ background: rgba(248,81,73,0.06); }}
tr.worst-row td:first-child {{ border-left: 3px solid var(--danger); }}
.filter-bar {{
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
    align-items: center;
}}
.filter-bar input[type="text"] {{
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 13px;
    width: 280px;
}}
.filter-bar input[type="text"]:focus {{ outline: none; border-color: var(--accent); }}
.filter-bar label {{ font-size: 12px; color: var(--text-muted); }}
.filter-bar select {{
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 13px;
}}
.note-type-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}}
.note-type-card {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}}
.note-type-card .nt-val {{
    font-size: 2rem;
    font-weight: 800;
    display: block;
}}
.note-type-card .nt-lbl {{
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}}
footer {{
    border-top: 1px solid var(--border);
    padding: 32px 24px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
}}
.text-green {{ color: var(--accent2); }}
.text-gold {{ color: var(--gold); }}
.text-red {{ color: var(--danger); }}
.text-muted {{ color: var(--text-muted); }}
.mt-4 {{ margin-top: 16px; }}
.mb-4 {{ margin-bottom: 16px; }}
</style>
</head>
<body>

<div class="live-banner">
  <div class="live-dot"></div>
  <span>&#10003; V8 MODEL COMPARISON REPORT</span>
  <small>Generated: {timestamp} &middot; {len(valid_results)} charts analyzed</small>
</div>

<header>
  <div class="header-tag">Pump It Up &mdash; Limb Annotation ML</div>
  <h1>v8 Model vs Old Model<br><span class="pct">{overall_v8_acc:.1f}%</span> Overall Accuracy</h1>
  <div class="header-stats">
    <div class="hstat gold"><span class="val">{overall_v8_acc:.1f}%</span><span class="lbl">v8 (MLX)</span></div>
    <div class="hstat"><span class="val" style="color:#9ca3af">{overall_old_acc:.1f}%</span><span class="lbl">Old Model</span></div>
    <div class="hstat green"><span class="val">{overall_v8_acc - overall_old_acc:+.1f}pp</span><span class="lbl">Delta</span></div>
    <div class="hstat"><span class="val" style="color:var(--accent2)">{v8_wins}/{total_cmp}</span><span class="lbl">Wins</span></div>
    <div class="hstat"><span class="val" style="color:var(--purple)">{avg_v8:.1f}%</span><span class="lbl">Avg v8</span></div>
  </div>
</header>

<main>

<section id="overview">
  <h2><span class="icon">&#128202;</span> Overall Accuracy Comparison</h2>

  <div class="card-grid">
    <div class="card">
      <h3>v8 Transformer (MLX)</h3>
      <div class="model-progress pbar-v8 mt-4">
        <div class="head">
          <span class="name">v8 — Singles Limb Model</span>
          <span class="pct-val" style="color:var(--gold)">{overall_v8_acc:.1f}%</span>
        </div>
        <div class="pbar-track"><div class="pbar-fill" style="width:{overall_v8_acc}%"></div></div>
      </div>
      <p style="margin-top:12px;color:var(--text-muted)">LimbSequenceTransformer with scheduled sampling. Causal mask, both-mirror augmentation. 14.7M params.</p>
    </div>
    <div class="card">
      <h3>Old LightGBM Model</h3>
      <div class="model-progress pbar-old mt-4">
        <div class="head">
          <span class="name">Old Model — LightGBM</span>
          <span class="pct-val" style="color:#9ca3af">{overall_old_acc:.1f}%</span>
        </div>
        <div class="pbar-track"><div class="pbar-fill" style="width:{overall_old_acc}%"></div></div>
      </div>
      <p style="margin-top:12px;color:var(--text-muted)">Previous generation model. Handcrafted features. No sequence modeling.</p>
    </div>
  </div>

  <div class="chart-wrap mt-4">
    <div class="chart-title">v8 vs Old Model — Tap Accuracy by Chart</div>
    <canvas id="compareChart" height="300"></canvas>
  </div>
</section>

<section id="note-types">
  <h2><span class="icon">&#127919;</span> Breakdown by Note Type</h2>
  <p style="color:var(--text-muted);margin-bottom:16px">v8 model accuracy broken down by note type: singles (isolated taps), jacks (rapid same-panel), triples (3+ simultaneous taps)</p>

  <div class="note-type-grid">
    <div class="note-type-card">
      <span class="nt-val" style="color:var(--accent)">{v8_single:.1f}%</span>
      <span class="nt-lbl">Single Notes</span>
    </div>
    <div class="note-type-card">
      <span class="nt-val" style="color:var(--accent3)">{v8_jack:.1f}%</span>
      <span class="nt-lbl">Jack Notes</span>
    </div>
    <div class="note-type-card">
      <span class="nt-val" style="color:var(--purple)">{v8_triple:.1f}%</span>
      <span class="nt-lbl">Triple Notes</span>
    </div>
  </div>

  <div class="chart-wrap mt-4">
    <div class="chart-title">Note Type Accuracy — v8 Model</div>
    <canvas id="noteTypeChart" height="200"></canvas>
  </div>
</section>

<section id="per-chart">
  <h2><span class="icon">&#128203;</span> Per-Chart Accuracy Table</h2>

  <div class="filter-bar">
    <input type="text" id="searchInput" placeholder="Search charts..." oninput="filterTable()">
    <label>Mode:</label>
    <select id="modeFilter" onchange="filterTable()">
      <option value="">All</option>
      <option value="S">Singles</option>
      <option value="D">Doubles</option>
    </select>
    <label>Group:</label>
    <select id="groupFilter" onchange="filterTable()">
      <option value="">All</option>
      <option value="worst_accuracy">Worst Accuracy</option>
      <option value="hardest_level">Hardest Level</option>
    </select>
  </div>

  <div class="card">
    <div style="overflow-x:auto;">
      <table id="chartTable">
        <thead>
          <tr>
            <th>Chart</th>
            <th>Song</th>
            <th>Mode</th>
            <th>Level</th>
            <th>Group</th>
            <th>Old Tap %</th>
            <th>v8 Tap %</th>
            <th>Delta</th>
            <th>v8 Jack %</th>
            <th>v8 Triple %</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>
</section>

<section id="top-charts">
  <h2><span class="icon">&#127942;</span> Best &amp; Worst Performing Charts</h2>
  <div class="card-grid">
    <div class="card success">
      <h3 class="text-green">Best Performing Chart</h3>
      <p><strong>{best.get('song_name', 'N/A')}</strong></p>
      <p>v8 Accuracy: <strong class="text-gold">{best.get('v8_tap', 0):.1f}%</strong></p>
      <p>Level: {best.get('level', 'N/A')} &middot; Mode: {best.get('mode', 'N/A')}</p>
      <p class="text-muted" style="margin-top:8px;font-size:12px">{best.get('shortname', 'N/A')}</p>
    </div>
    <div class="card danger">
      <h3 class="text-red">Worst Performing Chart</h3>
      <p><strong>{worst.get('song_name', 'N/A')}</strong></p>
      <p>v8 Accuracy: <strong class="text-red">{worst.get('v8_tap', 0):.1f}%</strong></p>
      <p>Level: {worst.get('level', 'N/A')} &middot; Mode: {worst.get('mode', 'N/A')}</p>
      <p class="text-muted" style="margin-top:8px;font-size:12px">{worst.get('shortname', 'N/A')}</p>
    </div>
  </div>
</section>

</main>

<footer>
  <p><strong>PIU Limb Annotation ML Project</strong> &mdash; piu-annotate_to_label_piu</p>
  <p style="margin-top:8px">v8 Model Comparison Report &middot; Generated: {timestamp}</p>
</footer>

<script>
const RESULTS = {results_json};

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "-apple-system, 'Segoe UI', sans-serif";

// Scatter chart: v8 vs old
const scatterData = RESULTS
    .filter(r => r.v8_tap !== null && r.old_tap !== null)
    .map(r => ({{ x: r.old_tap, y: r.v8_tap, label: r.shortname.substring(0, 30) }}));

new Chart(document.getElementById('compareChart'), {{
    type: 'scatter',
    data: {{
        datasets: [{{
            label: 'Charts',
            data: scatterData,
            backgroundColor: 'rgba(255,215,0,0.6)',
            borderColor: '#ffd700',
            borderWidth: 1,
            pointRadius: 5,
            pointHoverRadius: 7,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ display: false }},
            tooltip: {{
                backgroundColor: '#21262d',
                borderColor: '#30363d',
                borderWidth: 1,
                callbacks: {{
                    label: (ctx) => `${{ctx.raw.label}}: v8=${{ctx.raw.y.toFixed(1)}}% old=${{ctx.raw.x.toFixed(1)}}%`
                }}
            }}
        }},
        scales: {{
            x: {{
                title: {{ display: true, text: 'Old Model Accuracy (%)', font: {{ size: 11 }} }},
                min: 0, max: 100,
                grid: {{ color: 'rgba(48,54,61,0.5)' }}
            }},
            y: {{
                title: {{ display: true, text: 'v8 Model Accuracy (%)', font: {{ size: 11 }} }},
                min: 0, max: 100,
                grid: {{ color: 'rgba(48,54,61,0.5)' }}
            }}
        }}
    }}
}});

// Note type bar chart
new Chart(document.getElementById('noteTypeChart'), {{
    type: 'bar',
    data: {{
        labels: ['Single Notes', 'Jack Notes', 'Triple Notes'],
        datasets: [{{
            label: 'v8 Accuracy (%)',
            data: [{v8_single:.1f}, {v8_jack:.1f}, {v8_triple:.1f}],
            backgroundColor: ['rgba(88,166,255,0.7)', 'rgba(240,136,62,0.7)', 'rgba(163,113,247,0.7)'],
            borderColor: ['#58a6ff', '#f0883e', '#a371f7'],
            borderWidth: 2,
            borderRadius: 6,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            y: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }},
            x: {{ grid: {{ display: false }} }}
        }}
    }}
}});

// Table population
function populateTable(data) {{
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    data.forEach(r => {{
        const delta = r.v8_tap !== null && r.old_tap !== null
            ? (r.v8_tap - r.old_tap).toFixed(1)
            : 'n/a';
        const deltaNum = parseFloat(delta);
        const deltaColor = delta === 'n/a' ? '' : deltaNum >= 0 ? 'var(--accent2)' : 'var(--danger)';

        const isBest = r.v8_tap !== null && r.v8_tap >= 95;
        const isWorst = r.v8_tap !== null && r.v8_tap < 50;
        let rowClass = '';
        if (isBest) rowClass = 'best-row';
        else if (isWorst) rowClass = 'worst-row';

        const tr = document.createElement('tr');
        if (rowClass) tr.className = rowClass;
        tr.innerHTML = `
            <td style="font-size:12px">${{r.shortname?.substring(0, 40) ?? ''}}</td>
            <td>${{r.song_name ?? ''}}</td>
            <td>${{r.mode ?? ''}}</td>
            <td>${{r.level ?? ''}}</td>
            <td><span style="font-size:10px;background:var(--bg3);padding:2px 6px;border-radius:4px">${{(r.groups || []).join(', ')}}</span></td>
            <td>${{r.old_tap != null ? r.old_tap.toFixed(1) + '%' : 'n/a'}}</td>
            <td style="color:var(--gold)">${{r.v8_tap != null ? r.v8_tap.toFixed(1) + '%' : 'n/a'}}</td>
            <td style="color:${{deltaColor}}">${{delta === 'n/a' ? 'n/a' : (deltaNum >= 0 ? '+' : '') + delta + 'pp'}}</td>
            <td>${{r.v8_jack != null ? r.v8_jack.toFixed(1) + '%' : 'n/a'}}</td>
            <td>${{r.v8_triple != null ? r.v8_triple.toFixed(1) + '%' : 'n/a'}}</td>
        `;
        tbody.appendChild(tr);
    }});
}}

function filterTable() {{
    const search = document.getElementById('searchInput').value.toLowerCase();
    const mode = document.getElementById('modeFilter').value;
    const group = document.getElementById('groupFilter').value;

    const filtered = RESULTS.filter(r => {{
        const matchSearch = !search || ((r.shortname || '').toLowerCase().includes(search) || (r.song_name || '').toLowerCase().includes(search));
        const matchMode = !mode || r.mode === mode;
        const matchGroup = !group || ((r.groups || []).includes(group));
        return matchSearch && matchMode && matchGroup;
    }});
    populateTable(filtered);
}}

populateTable(RESULTS);
</script>
</body>
</html>'''
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--limit', type=int, default=0, help='Limit number of charts to process (0=all)')
    pargs = parser.parse_args()

    print(f'Loading benchmark from: {pargs.baseline}')
    baseline = json.load(open(pargs.baseline))
    if pargs.limit > 0:
        baseline = baseline[:pargs.limit]
    print(f'Processing {len(baseline)} benchmark charts...')

    setup_v8_model_args()
    vis_idx = load_vis_index(VIS_DIR)
    print(f'Indexed {len(vis_idx)} vis-ss charts')
    print(f'Loading v8 model from {MODEL_DIR}...')
    v8_model = create_v8_model(MODEL_DIR)
    print('v8 model loaded successfully!')

    from piu_annotate.ml import featurizers as ftz
    results = []

    for i, entry in enumerate(baseline):
        sn = entry['shortname']

        csv_path = find_csv_for_shortname(sn)
        ref_path = vis_idx.get(sn)

        result = {
            'shortname': sn,
            'song_name': entry['song_name'],
            'mode': entry['mode'],
            'level': entry['level'],
            'groups': entry['groups'],
            'old_tap': entry.get('tap_acc_old'),
            'old_jack': entry.get('jack_acc_old'),
            'old_triple': entry.get('triple_acc_old'),
            'v8_tap': None,
            'v8_jack': None,
            'v8_triple': None,
            'v8_single': None,
        }

        print(f'[{i+1}/{len(baseline)}] {sn[:50]}...')

        if not csv_path:
            print(f'  -> No CSV found, skipping')
            results.append(result)
            continue
        if not ref_path:
            print(f'  -> No vis-ss reference found, skipping')
            results.append(result)
            continue

        try:
            ref = json.load(open(ref_path))
            vis_arrows = ref[0]
        except Exception as e:
            print(f'  -> Failed to load vis-ss reference: {e}')
            results.append(result)
            continue

        try:
            cs = ChartStruct.from_file(csv_path)
            fcs = ftz.ChartStructFeaturizer(cs)

            # Get raw features for v8 model
            points = fcs.get_raw_features()
            pred_limbs = predict_v8(v8_model, points)

            labels = fcs.get_labels_from_limb_col('Limb annotation')
            pred_coords = fcs.pred_coords

            note_types = classify_note_type(pred_coords, cs)

            correct = sum(int(int(pred_limbs[i]) == int(labels[i])) for i in range(len(pred_limbs)))
            result['v8_tap'] = round(100.0 * correct / max(len(pred_limbs), 1), 2)

            by_type = {'single': {'correct': 0, 'total': 0},
                       'jack': {'correct': 0, 'total': 0},
                       'triple': {'correct': 0, 'total': 0}}
            for idx, nt in enumerate(note_types):
                by_type[nt]['total'] += 1
                by_type[nt]['correct'] += int(int(pred_limbs[idx]) == int(labels[idx]))

            for nt in by_type:
                d = by_type[nt]
                if d['total'] > 0:
                    result[f'v8_{nt}'] = round(100.0 * d['correct'] / d['total'], 2)

            print(f'  -> v8={result["v8_tap"]:.1f}% old={result["old_tap"]:.1f}% delta={result["v8_tap"] - result["old_tap"]:+.1f}pp')

        except Exception as e:
            import traceback
            print(f'  -> Inference failed: {e}')
            traceback.print_exc()

        results.append(result)

    print('\nComputing overall accuracy...')
    valid_results = [r for r in results if r['v8_tap'] is not None]
    overall_v8 = sum(r['v8_tap'] for r in valid_results) / len(valid_results) if valid_results else 0

    old_valid = [r for r in valid_results if r['old_tap'] is not None]
    overall_old = sum(r['old_tap'] for r in old_valid) / len(old_valid) if old_valid else 0

    print(f'Overall v8 accuracy: {overall_v8:.2f}%')
    print(f'Overall old accuracy: {overall_old:.2f}%')

    print('Generating HTML report...')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = build_report(results, overall_v8, overall_old, timestamp)

    os.makedirs(os.path.dirname(pargs.output), exist_ok=True)
    with open(pargs.output, 'w') as f:
        f.write(html)
    print(f'Report saved to: {pargs.output}')


if __name__ == '__main__':
    main()
