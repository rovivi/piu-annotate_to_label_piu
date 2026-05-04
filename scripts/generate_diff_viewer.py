#!/usr/bin/env python3
"""
generate_diff_viewer.py

Genera un visor HTML standalone que compara anotaciones del modelo (processed_db)
contra la referencia vis-ss ground truth, nota por nota.

Features:
  - Navegación entre discrepancias estilo VS Code (F8 / Shift+F8 o botones ▲▼)
  - Búsqueda de charts en tiempo real (Ctrl+F)
  - Minimap de errores en cada panel (como el scrollbar de VS Code)
  - Scroll sincronizado entre paneles
  - Zoom con Shift+Scroll

Usage:
    python3 scripts/generate_diff_viewer.py                 # 40 peores singles + doubles
    python3 scripts/generate_diff_viewer.py --mode S        # solo singles
    python3 scripts/generate_diff_viewer.py --n_charts 60 --mode all
    python3 scripts/generate_diff_viewer.py --song "Like Me"
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

VIS_DIR  = '/Users/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524'
PROC_DIR = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/processed_db'
IMG_SRC  = '/Users/rodrigo/dev/piu/piulatam/public/images/arrows-hint'
OUT_DIR  = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/diff_viewer'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_vis_index(vis_dir: str) -> dict[str, list]:
    idx = {}
    for fn in os.listdir(vis_dir):
        if not fn.endswith('.json'):
            continue
        try:
            data = json.load(open(os.path.join(vis_dir, fn)))
            sn = data[2].get('shortname')
            if sn:
                idx[sn] = data
        except Exception:
            pass
    return idx


# ── Diff & stats ──────────────────────────────────────────────────────────────

def _is_wrong(ref_limb: str, pred_limb: str) -> bool:
    r = (ref_limb  or 'e')[0]
    p = (pred_limb or 'e')[0]
    if r in ('e', '?') or p in ('e', '?'):
        return False
    return r != p


def compute_diffs(ref_cjs: list, pred_cjs: list) -> dict:
    tap_diffs  = [int(_is_wrong(r[2], p[2])) for r, p in zip(ref_cjs[0], pred_cjs[0])]
    hold_diffs = [int(_is_wrong(r[3], p[3])) for r, p in zip(ref_cjs[1], pred_cjs[1])]
    return {'taps': tap_diffs, 'holds': hold_diffs}


def compute_stats(ref_cjs: list, pred_cjs: list) -> dict | None:
    ref_taps,  pred_taps  = ref_cjs[0],  pred_cjs[0]
    ref_holds, pred_holds = ref_cjs[1],  pred_cjs[1]
    if len(ref_taps) != len(pred_taps):
        return None

    times = Counter(round(t[1], 5) for t in ref_taps)
    st = defaultdict(int)

    for i, (r, p) in enumerate(zip(ref_taps, pred_taps)):
        t   = round(r[1], 5)
        cnt = times[t]
        ok  = not _is_wrong(r[2], p[2])
        st['tap_total']   += 1
        st['tap_correct'] += int(ok)
        if cnt >= 3:
            st['triple_total']   += 1
            st['triple_correct'] += int(ok)
        if i > 0 and r[0] == ref_taps[i-1][0] and abs(r[1] - ref_taps[i-1][1]) > 1e-3:
            st['jack_total']   += 1
            st['jack_correct'] += int(ok)

    for r, p in zip(ref_holds, pred_holds):
        ok = not _is_wrong(r[3], p[3])
        st['hold_total']   += 1
        st['hold_correct'] += int(ok)

    def pct(n, d): return round(100 * n / d, 1) if d > 0 else None
    return {
        'tap':    pct(st['tap_correct'],    st['tap_total']),
        'triple': pct(st['triple_correct'], st['triple_total']),
        'jack':   pct(st['jack_correct'],   st['jack_total']),
        'hold':   pct(st['hold_correct'],   st['hold_total']),
        'tap_total': int(st['tap_total']),
        'tap_wrong': int(st['tap_total'] - st['tap_correct']),
    }


# ── Chart collection ──────────────────────────────────────────────────────────

def build_chart_data(
    proc_dir: str,
    vis_idx: dict,
    n_charts: int,
    mode_filter: str,
    song_filter: str | None,
) -> list[dict]:
    charts = []
    for fname in sorted(os.listdir(proc_dir)):
        if not fname.endswith('.json'):
            continue
        try:
            proc = json.load(open(os.path.join(proc_dir, fname)))
        except Exception:
            continue

        mode = proc.get('mode', '')
        if mode_filter != 'all' and mode != mode_filter:
            continue

        song_name = proc.get('song_name', '')
        if song_filter and song_filter.lower() not in song_name.lower():
            continue

        meta      = proc['cjs'][2]
        shortname = meta.get('shortname', '')
        ref       = vis_idx.get(shortname)
        if not ref:
            continue

        stats = compute_stats(ref, proc['cjs'])
        if not stats or stats['tap'] is None:
            continue

        diffs = compute_diffs(ref, proc['cjs'])
        charts.append({
            'id':    shortname,
            'name':  song_name,
            'level': proc.get('level', 0),
            'mode':  mode,
            'stats': stats,
            'ref':   ref,
            'pred':  proc['cjs'],
            'diffs': diffs,
        })

    charts.sort(key=lambda c: c['stats']['tap'])
    return charts[:n_charts]


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIU Limb Diff Viewer</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#0e0e0e; --bg2:#161616; --bg3:#1e1e1e; --border:#2a2a2a;
  --text:#d4d4d4; --muted:#555; --accent:#3b82f6;
  --wrong:#ef4444; --ok:#22c55e; --focus:#fbbf24;
  --sidebar-w:256px; --header-h:48px; --label-h:28px; --minimap-w:12px;
}
html,body { height:100%; overflow:hidden; background:var(--bg); color:var(--text);
            font:13px/1.4 system-ui,sans-serif; }
#app { display:flex; height:100%; }

/* ── Sidebar ── */
#sidebar {
  width:var(--sidebar-w); min-width:var(--sidebar-w); background:var(--bg2);
  border-right:1px solid var(--border); display:flex; flex-direction:column; overflow:hidden;
}
#sidebar-top { padding:10px 10px 0; }
#sidebar-title { font-size:10px; font-weight:700; color:var(--muted); text-transform:uppercase;
                 letter-spacing:.1em; margin-bottom:8px; }
#search-wrap { position:relative; margin-bottom:8px; }
#search-input {
  width:100%; padding:5px 8px 5px 26px; background:var(--bg3); border:1px solid var(--border);
  border-radius:4px; color:var(--text); font-size:12px; outline:none;
}
#search-input:focus { border-color:var(--accent); }
#search-icon { position:absolute; left:8px; top:50%; transform:translateY(-50%);
               color:var(--muted); font-size:11px; pointer-events:none; }
#search-clear { position:absolute; right:6px; top:50%; transform:translateY(-50%);
                color:var(--muted); font-size:14px; cursor:pointer; display:none; }
#mode-tabs { display:flex; gap:3px; margin-bottom:8px; }
.mode-tab {
  flex:1; padding:3px 0; text-align:center; border-radius:3px; font-size:11px;
  font-weight:600; cursor:pointer; color:var(--muted); background:var(--bg3);
  border:1px solid var(--border); transition:all .15s;
}
.mode-tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
#chart-count { font-size:10px; color:var(--muted); padding:0 10px 6px; }

#chart-list { flex:1; overflow-y:auto; }
.chart-item {
  padding:7px 10px; cursor:pointer; border-bottom:1px solid var(--border);
  border-left:3px solid transparent;
}
.chart-item:hover { background:var(--bg3); }
.chart-item.active { background:#1a2535; border-left-color:var(--accent); }
.chart-item.hidden { display:none; }
.ci-name { font-size:11px; font-weight:600; line-height:1.3; margin-bottom:2px;
           white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ci-meta { display:flex; justify-content:space-between; font-size:10px; color:var(--muted); }
.ci-acc { font-size:10px; font-weight:700; }
.ci-bar { height:2px; border-radius:1px; margin-top:3px; background:var(--bg3); }
.ci-fill { height:100%; border-radius:1px; }

/* ── Main ── */
#main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
#header {
  height:var(--header-h); background:var(--bg2); border-bottom:1px solid var(--border);
  display:flex; align-items:center; padding:0 12px; gap:12px; flex-shrink:0;
}
#chart-title { font-size:12px; font-weight:700; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* diff nav */
#diff-nav { display:flex; align-items:center; gap:5px; flex-shrink:0; }
.nav-btn {
  padding:3px 8px; background:var(--bg3); border:1px solid var(--border);
  border-radius:3px; color:var(--text); font-size:11px; cursor:pointer;
  display:flex; align-items:center; gap:3px;
}
.nav-btn:hover { background:#2a2a2a; border-color:#444; }
#diff-counter { font-size:11px; color:var(--muted); white-space:nowrap; min-width:50px; text-align:center; }

/* stats */
#stats-bar { display:flex; gap:10px; font-size:10px; flex-shrink:0; }
.stat-item { display:flex; flex-direction:column; align-items:center; gap:1px; }
.stat-label { color:var(--muted); font-size:8px; text-transform:uppercase; }
.stat-value { font-weight:700; font-size:11px; }
.sg { color:var(--ok); } .sm { color:#f59e0b; } .sb { color:var(--wrong); }

/* ── Panels ── */
#panels { flex:1; display:flex; overflow:hidden; }
.panel-wrap { flex:1; display:flex; flex-direction:column; overflow:hidden; min-width:0; position:relative; }
.panel-wrap + .panel-wrap { border-left:2px solid var(--border); }
.panel-label {
  height:var(--label-h); background:var(--bg3); border-bottom:1px solid var(--border);
  display:flex; align-items:center; padding:0 10px; font-size:10px; font-weight:700;
  flex-shrink:0; gap:6px; text-transform:uppercase; letter-spacing:.05em;
}
.label-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.dot-ref  { background:#22c55e; }
.dot-pred { background:#f59e0b; }

.canvas-area { flex:1; position:relative; overflow:hidden; }
#left-canvas, #right-canvas {
  position:absolute; top:0; left:0; display:block; image-rendering:pixelated;
}
/* minimap */
.minimap {
  position:absolute; top:0; right:0; width:var(--minimap-w); height:100%;
  cursor:pointer; z-index:5;
}
.scroll-driver {
  position:absolute; top:0; left:0; height:100%; overflow-y:scroll;
  opacity:0; z-index:10; cursor:default;
}
.scroll-spacer { width:1px; }

/* ── Footer ── */
#footer {
  padding:4px 12px; background:var(--bg2); border-top:1px solid var(--border);
  display:flex; align-items:center; gap:16px; font-size:9px; color:var(--muted); flex-shrink:0;
}
.leg { display:flex; align-items:center; gap:4px; }
.lc { width:10px; height:10px; border-radius:50%; border:2px solid; }
.lc-w { border-color:var(--wrong); }
.lc-f { border-color:var(--focus); background:transparent; }

kbd {
  padding:1px 4px; background:var(--bg3); border:1px solid var(--border);
  border-radius:2px; font-family:monospace; font-size:9px;
}

/* loading */
#loading {
  position:absolute; inset:0; display:none; align-items:center; justify-content:center;
  background:rgba(14,14,14,.85); z-index:30; font-size:12px; color:var(--muted);
  flex-direction:column; gap:10px;
}
.spinner { width:24px; height:24px; border:2px solid var(--border);
           border-top-color:var(--accent); border-radius:50%; animation:spin .7s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<div id="app">

<!-- Sidebar ─────────────────────────────────────────────────────────────── -->
<div id="sidebar">
  <div id="sidebar-top">
    <div id="sidebar-title">PIU Diff Viewer</div>
    <div id="search-wrap">
      <span id="search-icon">⌕</span>
      <input id="search-input" type="text" placeholder="Buscar chart… (Ctrl+F)"
             oninput="onSearch(this.value)" autocomplete="off" spellcheck="false">
      <span id="search-clear" onclick="clearSearch()">✕</span>
    </div>
    <div id="mode-tabs">
      <div class="mode-tab active" onclick="filterMode(this,'all')">Todo</div>
      <div class="mode-tab" onclick="filterMode(this,'S')">Singles</div>
      <div class="mode-tab" onclick="filterMode(this,'D')">Doubles</div>
    </div>
  </div>
  <div id="chart-count"></div>
  <div id="chart-list"></div>
</div>

<!-- Main ────────────────────────────────────────────────────────────────── -->
<div id="main">
  <div id="header">
    <div id="chart-title">← Selecciona un chart</div>

    <div id="diff-nav" style="display:none">
      <button class="nav-btn" onclick="prevDiff()" title="Diff anterior  Shift+F8 / Alt+↑">▲ <span>Prev</span></button>
      <span id="diff-counter">— / —</span>
      <button class="nav-btn" onclick="nextDiff()" title="Siguiente diff  F8 / Alt+↓">▼ <span>Next</span></button>
    </div>

    <div id="stats-bar" style="display:none">
      <div class="stat-item"><div class="stat-label">Taps</div><div class="stat-value" id="s-tap">—</div></div>
      <div class="stat-item"><div class="stat-label">Triples</div><div class="stat-value" id="s-tri">—</div></div>
      <div class="stat-item"><div class="stat-label">Jacks</div><div class="stat-value" id="s-jck">—</div></div>
      <div class="stat-item"><div class="stat-label">Holds</div><div class="stat-value" id="s-hld">—</div></div>
      <div class="stat-item"><div class="stat-label">Errores</div><div class="stat-value sb" id="s-err">—</div></div>
    </div>
  </div>

  <div id="panels">
    <!-- Reference panel -->
    <div class="panel-wrap">
      <div class="panel-label"><div class="label-dot dot-ref"></div>Referencia (vis-ss)</div>
      <div class="canvas-area" id="left-area">
        <canvas id="left-canvas"></canvas>
        <canvas class="minimap" id="left-mini"></canvas>
        <div class="scroll-driver" id="left-scroll">
          <div class="scroll-spacer" id="left-spacer"></div>
        </div>
      </div>
    </div>

    <!-- Prediction panel -->
    <div class="panel-wrap">
      <div class="panel-label"><div class="label-dot dot-pred"></div>Predicción modelo</div>
      <div class="canvas-area" id="right-area">
        <canvas id="right-canvas"></canvas>
        <canvas class="minimap" id="right-mini"></canvas>
        <div class="scroll-driver" id="right-scroll">
          <div class="scroll-spacer" id="right-spacer"></div>
        </div>
      </div>
    </div>
  </div>

  <div id="footer">
    <div class="leg"><div class="lc lc-w"></div>Error del modelo</div>
    <div class="leg"><div class="lc lc-f"></div>Diff enfocado</div>
    <span>Pie izq <span style="color:#3b82f6">●</span> Pie der <span style="color:#ec4899">●</span> Cualquiera <span style="color:#f59e0b">●</span></span>
    <span style="margin-left:auto">
      <kbd>F8</kbd> siguiente · <kbd>Shift+F8</kbd> anterior ·
      <kbd>Shift+Scroll</kbd> zoom · <kbd>Ctrl+F</kbd> buscar
    </span>
  </div>
</div>

<div id="loading"><div class="spinner"></div>Cargando imágenes…</div>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────────
const IMG_DIR = '/*IMG_DIR*/';
const CHARTS  = /*CHARTS_DATA*/;

// ── Image cache ───────────────────────────────────────────────────────────────
const PANEL_DIRS = ['downleft','upleft','center','upright','downright'];
const LIMB_NAMES = {l:'left',r:'right',e:'either',h:'hand','?':'either'};

const ALL_IMG_NAMES = (() => {
  const names = [];
  for (const p of ['arrow','trail','holdcap'])
    for (const d of PANEL_DIRS)
      for (const l of ['left','right','either','hand'])
        names.push(`${p}_${d}_${l}.png`);
  return names;
})();

let imgCache = {};

async function preloadImages() {
  document.getElementById('loading').style.display = 'flex';
  await Promise.all(ALL_IMG_NAMES.map(n => new Promise(res => {
    const img = new Image();
    img.onload = () => { imgCache[n] = img; res(); };
    img.onerror = res;
    img.src = IMG_DIR + '/' + n;
  })));
  document.getElementById('loading').style.display = 'none';
}

function getImg(panel, limb, type) {
  const dir = PANEL_DIRS[panel % 5];
  const l   = LIMB_NAMES[(limb || 'e')[0]] ?? 'either';
  const p   = type === 'arrow' ? 'arrow' : type === 'trail' ? 'trail' : 'holdcap';
  return imgCache[`${p}_${dir}_${l}.png`];
}

// ── Layout constants ──────────────────────────────────────────────────────────
const ARROWS_X = 28;
const PANEL_W  = 40;
const ARROW_SZ = 40;
const MINI_W   = 12;

// ── State ─────────────────────────────────────────────────────────────────────
let pxPerSec      = 320;
let scrollY       = 0;
let currentChart  = null;
let rafId         = null;
let diffTimes     = [];   // [{time, idx}] sorted
let diffIdx       = -1;   // which diff is focused
let focusedTime   = -1;

// ── Helpers ───────────────────────────────────────────────────────────────────
function numPanels(cjs) {
  const maxP = Math.max(...cjs[0].map(t=>t[0]), ...cjs[1].map(h=>h[0]), 4);
  return maxP >= 5 ? 10 : 5;
}

function chartTotalSecs(chart) {
  const c = chart.ref;
  return Math.max(
    c[0].length ? c[0][c[0].length-1][1] : 0,
    c[1].length ? c[1][c[1].length-1][2] : 0
  );
}
function chartHeight(chart) { return chartTotalSecs(chart) * pxPerSec + 200; }

// ── Rendering ─────────────────────────────────────────────────────────────────
function drawGrid(ctx, W, H, t0, t1, nP) {
  const gridEnd = ARROWS_X + nP * PANEL_W;
  ctx.fillStyle = '#111'; ctx.fillRect(0,0,W,H);

  ctx.strokeStyle = '#1d1d1d'; ctx.lineWidth = 1;
  for (let s = Math.floor(t0); s <= Math.ceil(t1)+1; s++) {
    const y = (s-t0)*pxPerSec;
    ctx.beginPath(); ctx.moveTo(ARROWS_X,y); ctx.lineTo(gridEnd,y); ctx.stroke();
  }
  ctx.fillStyle = '#3a3a3a'; ctx.font = '9px monospace';
  for (let s = Math.floor(t0/5)*5; s <= Math.ceil(t1)+5; s+=5) {
    const y = (s-t0)*pxPerSec;
    const m = Math.floor(s/60), sc = String(s%60).padStart(2,'0');
    ctx.fillText(`${m}:${sc}`, 2, y+3);
  }
  ctx.strokeStyle = '#252525'; ctx.lineWidth = 1;
  for (let p=0; p<=nP; p++) {
    const x = ARROWS_X + p*PANEL_W;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
  }
  if (nP===10) {
    ctx.strokeStyle='#2e2e2e'; ctx.lineWidth=2;
    const x = ARROWS_X+5*PANEL_W;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
  }
}

function drawPanel(canvasEl, cjs, diffs, isRef) {
  if (!canvasEl || !cjs) return;
  const ctx = canvasEl.getContext('2d');
  const W=canvasEl.width, H=canvasEl.height;
  const t0=scrollY/pxPerSec, t1=t0+H/pxPerSec;
  const nP=numPanels(cjs);

  drawGrid(ctx, W, H, t0, t1, nP);

  const taps=cjs[0], holds=cjs[1];
  const tapDiff=diffs?.taps??[], holdDiff=diffs?.holds??[];

  // trails + caps
  for (let i=0; i<holds.length; i++) {
    const [panel,s,e,limb]=holds[i];
    if (e<t0-0.5 || s>t1+0.5) continue;
    const x=ARROWS_X+panel*PANEL_W;
    const yS=(s-t0)*pxPerSec, yE=(e-t0)*pxPerSec;
    const ti=getImg(panel,limb,'trail');
    if (ti) ctx.drawImage(ti, x, yS+ARROW_SZ/2, ARROW_SZ, Math.max(yE-yS,2));
    const ci=getImg(panel,limb,'cap');
    if (ci) ctx.drawImage(ci, x, yE, ARROW_SZ, ARROW_SZ);
  }

  function drawNote(panel, time, limb, isWrong) {
    const x=ARROWS_X+panel*PANEL_W, y=(time-t0)*pxPerSec;
    if (y<-ARROW_SZ||y>H+ARROW_SZ) return;
    const img=getImg(panel,limb,'arrow');
    if (img) ctx.drawImage(img,x,y,ARROW_SZ,ARROW_SZ);
    const cx=x+ARROW_SZ/2, cy=y+ARROW_SZ/2;

    // focused diff highlight (both panels)
    const isFocused = focusedTime >= 0 && Math.abs(time - focusedTime) < 0.06;
    if (isFocused) {
      ctx.beginPath();
      ctx.arc(cx, cy, ARROW_SZ/2+5, 0, Math.PI*2);
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.stroke();
      // outer glow
      ctx.beginPath();
      ctx.arc(cx, cy, ARROW_SZ/2+9, 0, Math.PI*2);
      ctx.strokeStyle = 'rgba(251,191,36,0.2)';
      ctx.lineWidth = 4;
      ctx.stroke();
    }

    if (!isRef && isWrong) {
      // solid red ring on prediction
      ctx.beginPath();
      ctx.arc(cx, cy, ARROW_SZ/2-1, 0, Math.PI*2);
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 3; ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, ARROW_SZ/2+4, 0, Math.PI*2);
      ctx.strokeStyle = 'rgba(239,68,68,0.2)'; ctx.lineWidth = 5; ctx.stroke();
    }
    if (isRef && isWrong) {
      // dashed ring on reference
      ctx.beginPath();
      ctx.arc(cx, cy, ARROW_SZ/2-1, 0, Math.PI*2);
      ctx.strokeStyle='rgba(239,68,68,0.45)'; ctx.lineWidth=2;
      ctx.setLineDash([4,3]); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  for (let i=0;i<holds.length;i++) {
    const [panel,s,,limb]=holds[i];
    drawNote(panel,s,limb,holdDiff[i]===1);
  }
  for (let i=0;i<taps.length;i++) {
    const [panel,time,limb]=taps[i];
    drawNote(panel,time,limb,tapDiff[i]===1);
  }
}

// ── Minimap ───────────────────────────────────────────────────────────────────
function drawMinimap(canvasEl, chart) {
  if (!canvasEl || !chart) return;
  const ctx=canvasEl.getContext('2d');
  const W=canvasEl.width, H=canvasEl.height;
  const totalSecs=chartTotalSecs(chart);
  if (totalSecs<=0) return;

  ctx.fillStyle='#151515'; ctx.fillRect(0,0,W,H);

  const diffs=chart.diffs, cjs=chart.ref;

  // draw each tap diff as a red pixel-row
  for (let i=0;i<diffs.taps.length;i++) {
    if (!diffs.taps[i]) continue;
    const t=cjs[0][i][1];
    const y=Math.round((t/totalSecs)*H);
    ctx.fillStyle='rgba(239,68,68,0.8)';
    ctx.fillRect(0, Math.max(0,y-1), W, 2);
  }
  for (let i=0;i<diffs.holds.length;i++) {
    if (!diffs.holds[i]) continue;
    const t=cjs[1][i][1];
    const y=Math.round((t/totalSecs)*H);
    ctx.fillStyle='rgba(239,68,68,0.6)';
    ctx.fillRect(0, Math.max(0,y-1), W, 2);
  }

  // focused diff marker
  if (focusedTime>=0) {
    const y=Math.round((focusedTime/totalSecs)*H);
    ctx.fillStyle='#fbbf24';
    ctx.fillRect(0, Math.max(0,y-2), W, 4);
  }

  // viewport indicator
  const ch=chartHeight(chart);
  const vpTop=Math.round((scrollY/ch)*H);
  const vpH=Math.max(6, Math.round((getCanvasH()/ch)*H));
  ctx.fillStyle='rgba(255,255,255,0.07)'; ctx.fillRect(0,vpTop,W,vpH);
  ctx.strokeStyle='rgba(255,255,255,0.25)'; ctx.lineWidth=1;
  ctx.strokeRect(0.5,vpTop+0.5,W-1,vpH-1);
}

function getCanvasH() {
  return document.getElementById('left-area')?.clientHeight ?? window.innerHeight;
}

function render() {
  if (!currentChart) return;
  const lc=document.getElementById('left-canvas');
  const rc=document.getElementById('right-canvas');
  const lm=document.getElementById('left-mini');
  const rm=document.getElementById('right-mini');
  drawPanel(lc, currentChart.ref,  currentChart.diffs, true);
  drawPanel(rc, currentChart.pred, currentChart.diffs, false);
  drawMinimap(lm, currentChart);
  drawMinimap(rm, currentChart);
}

// ── Scroll ────────────────────────────────────────────────────────────────────
let isSyncing=false;

function setupScroll() {
  const ls=document.getElementById('left-scroll');
  const rs=document.getElementById('right-scroll');

  function onScroll(e) {
    if (isSyncing) return;
    isSyncing=true;
    const sy=e.target.scrollTop;
    scrollY=sy;
    ls.scrollTop=sy; rs.scrollTop=sy;
    cancelAnimationFrame(rafId);
    rafId=requestAnimationFrame(render);
    isSyncing=false;
  }

  ls.addEventListener('scroll', onScroll, {passive:true});
  rs.addEventListener('scroll', onScroll, {passive:true});

  // Shift+scroll = zoom
  for (const el of [ls, rs]) {
    el.addEventListener('wheel', e=>{
      if (!e.shiftKey) return;
      e.preventDefault();
      const d=e.deltaY>0?0.85:1.15;
      pxPerSec=Math.max(80,Math.min(900,pxPerSec*d));
      if (currentChart) { updateSpacers(); }
      requestAnimationFrame(render);
    }, {passive:false});
  }

  // Minimap click → jump
  for (const id of ['left-mini','right-mini']) {
    document.getElementById(id).addEventListener('click', e=>{
      if (!currentChart) return;
      const rect=e.currentTarget.getBoundingClientRect();
      const frac=(e.clientY-rect.top)/rect.height;
      const targetY=frac*chartHeight(currentChart);
      smoothScroll(targetY);
    });
  }
}

// smooth animated scroll
let smoothAnim=null;
function smoothScroll(targetY) {
  if (smoothAnim) cancelAnimationFrame(smoothAnim);
  const startY=scrollY, dist=targetY-startY;
  if (Math.abs(dist)<2) return;
  const dur=Math.min(650, 180+Math.abs(dist)*0.25);
  const t0=performance.now();

  function step(now) {
    const p=Math.min((now-t0)/dur, 1);
    const ease=1-Math.pow(1-p,3);
    const newY=Math.max(0,startY+dist*ease);
    scrollY=newY;
    const ls=document.getElementById('left-scroll');
    const rs=document.getElementById('right-scroll');
    ls.scrollTop=newY; rs.scrollTop=newY;
    requestAnimationFrame(render);
    if (p<1) smoothAnim=requestAnimationFrame(step);
    else smoothAnim=null;
  }
  smoothAnim=requestAnimationFrame(step);
}

function updateSpacers() {
  const h=chartHeight(currentChart)+'px';
  document.getElementById('left-spacer').style.height=h;
  document.getElementById('right-spacer').style.height=h;
}

// ── Canvas sizing ─────────────────────────────────────────────────────────────
function resizeCanvases() {
  for (const side of ['left','right']) {
    const area=document.getElementById(`${side}-area`);
    const cvs=document.getElementById(`${side}-canvas`);
    const mini=document.getElementById(`${side}-mini`);
    const aW=area.clientWidth, aH=area.clientHeight;
    const nP=currentChart ? numPanels(currentChart.ref) : 5;
    const chartW=Math.min(aW-MINI_W-2, ARROWS_X+nP*PANEL_W+20);
    if (cvs.width!==chartW||cvs.height!==aH) { cvs.width=chartW; cvs.height=aH; }
    if (mini.width!==MINI_W||mini.height!==aH) { mini.width=MINI_W; mini.height=aH; }
    // scroll driver covers full area
    const drv=document.getElementById(`${side}-scroll`);
    drv.style.width=aW+'px'; drv.style.height=aH+'px';
  }
}

// ── Diff navigation ───────────────────────────────────────────────────────────
function buildDiffTimes(chart) {
  const diffs=chart.diffs, cjs=chart.ref;
  const raw=[];
  for (let i=0;i<diffs.taps.length;i++)
    if (diffs.taps[i]) raw.push(cjs[0][i][1]);
  for (let i=0;i<diffs.holds.length;i++)
    if (diffs.holds[i]) raw.push(cjs[1][i][1]);
  raw.sort((a,b)=>a-b);
  // deduplicate within 60ms window
  const out=[];
  for (const t of raw)
    if (!out.length || t-out[out.length-1]>0.06) out.push(t);
  return out;
}

function updateDiffCounter() {
  const el=document.getElementById('diff-counter');
  if (!diffTimes.length) { el.textContent='sin errores'; return; }
  el.textContent=`${diffIdx+1} / ${diffTimes.length}`;
}

function goToDiff(idx) {
  if (!diffTimes.length) return;
  diffIdx=((idx%diffTimes.length)+diffTimes.length)%diffTimes.length;
  focusedTime=diffTimes[diffIdx];
  const canvasH=getCanvasH();
  const targetY=Math.max(0, focusedTime*pxPerSec - canvasH/2 + ARROW_SZ);
  smoothScroll(targetY);
  updateDiffCounter();
}

function nextDiff() { goToDiff(diffIdx+1); }
function prevDiff() { goToDiff(diffIdx-1); }

// ── Chart selection ───────────────────────────────────────────────────────────
function selectChart(chartId) {
  const chart=CHARTS.find(c=>c.id===chartId);
  if (!chart) return;
  currentChart=chart;
  scrollY=0; diffIdx=-1; focusedTime=-1;

  document.querySelectorAll('.chart-item').forEach(el=>
    el.classList.toggle('active', el.dataset.id===chartId)
  );
  // scroll sidebar item into view
  const active=document.querySelector('.chart-item.active');
  if (active) active.scrollIntoView({block:'nearest'});

  document.getElementById('chart-title').textContent=
    `${chart.name}  ·  ${chart.mode}${chart.level}  ·  ${chart.stats.tap_wrong} errores`;

  // stats
  const statsBar=document.getElementById('stats-bar');
  statsBar.style.display='flex';
  function setStat(id,v,hi,mid) {
    const el=document.getElementById(id);
    if (v==null){el.textContent='n/a';el.className='stat-value';return;}
    el.textContent=v.toFixed(1)+'%';
    el.className='stat-value '+(v>=hi?'sg':v>=mid?'sm':'sb');
  }
  setStat('s-tap',chart.stats.tap,    95,85);
  setStat('s-tri',chart.stats.triple, 90,75);
  setStat('s-jck',chart.stats.jack,   92,80);
  setStat('s-hld',chart.stats.hold,   97,90);
  document.getElementById('s-err').textContent=chart.stats.tap_wrong+' tap(s)';

  // diff nav
  diffTimes=buildDiffTimes(chart);
  document.getElementById('diff-nav').style.display=diffTimes.length?'flex':'none';
  updateDiffCounter();

  // reset scroll + size
  document.getElementById('left-scroll').scrollTop=0;
  document.getElementById('right-scroll').scrollTop=0;
  resizeCanvases();
  updateSpacers();
  requestAnimationFrame(render);

  // auto-jump to first diff after a short delay
  if (diffTimes.length) setTimeout(()=>goToDiff(0), 120);
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
let currentMode='all';
let searchQuery='';

function accColor(v) {
  if (v==null) return '#555';
  return v>=95?'#22c55e':v>=85?'#f59e0b':'#ef4444';
}

function populateSidebar() {
  const list=document.getElementById('chart-list');
  let visible=0;
  list.querySelectorAll('.chart-item').forEach(el=>{
    const c=CHARTS.find(c=>c.id===el.dataset.id);
    if (!c) return;
    const modeOk=currentMode==='all'||c.mode===currentMode;
    const searchOk=!searchQuery||c.name.toLowerCase().includes(searchQuery);
    const show=modeOk&&searchOk;
    el.classList.toggle('hidden',!show);
    if (show) visible++;
  });
  document.getElementById('chart-count').textContent=`${visible} chart${visible!==1?'s':''}`;
}

function buildSidebar() {
  const list=document.getElementById('chart-list');
  list.innerHTML='';
  CHARTS.forEach(c=>{
    const tap=c.stats.tap??0;
    const col=accColor(tap);
    const div=document.createElement('div');
    div.className='chart-item';
    div.dataset.id=c.id;
    div.onclick=()=>selectChart(c.id);
    div.innerHTML=`
      <div class="ci-name" title="${c.name}">${c.name}</div>
      <div class="ci-meta">
        <span>${c.mode}${c.level}</span>
        <span style="color:${col}">${tap!=null?tap.toFixed(1)+'%':'n/a'}</span>
        <span>${c.stats.tap_wrong} err</span>
      </div>
      <div class="ci-bar"><div class="ci-fill" style="width:${tap}%;background:${col}"></div></div>`;
    list.appendChild(div);
  });
  populateSidebar();
}

function filterMode(btn, mode) {
  currentMode=mode;
  document.querySelectorAll('.mode-tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  populateSidebar();
}

function onSearch(val) {
  searchQuery=val.toLowerCase().trim();
  document.getElementById('search-clear').style.display=val?'block':'none';
  populateSidebar();
}
function clearSearch() {
  const inp=document.getElementById('search-input');
  inp.value=''; onSearch(''); inp.focus();
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e=>{
  // F8 / Shift+F8
  if (e.key==='F8') {
    e.preventDefault();
    e.shiftKey ? prevDiff() : nextDiff();
    return;
  }
  // Alt+↓ / Alt+↑
  if (e.altKey && e.key==='ArrowDown') { e.preventDefault(); nextDiff(); return; }
  if (e.altKey && e.key==='ArrowUp')   { e.preventDefault(); prevDiff(); return; }
  // Ctrl+F → focus search
  if ((e.ctrlKey||e.metaKey) && e.key==='f') {
    e.preventDefault();
    const inp=document.getElementById('search-input');
    inp.focus(); inp.select();
    return;
  }
  // Escape → clear search
  if (e.key==='Escape') {
    const inp=document.getElementById('search-input');
    if (document.activeElement===inp) { clearSearch(); inp.blur(); }
    return;
  }
  // ← → switch charts
  if (e.key==='ArrowRight' && !e.altKey && document.activeElement.tagName!=='INPUT') {
    const items=visibleChartIds();
    const idx=items.indexOf(currentChart?.id??'');
    if (idx<items.length-1) selectChart(items[idx+1]);
    return;
  }
  if (e.key==='ArrowLeft' && !e.altKey && document.activeElement.tagName!=='INPUT') {
    const items=visibleChartIds();
    const idx=items.indexOf(currentChart?.id??'');
    if (idx>0) selectChart(items[idx-1]);
    return;
  }
});

function visibleChartIds() {
  return [...document.querySelectorAll('.chart-item:not(.hidden)')].map(el=>el.dataset.id);
}

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async ()=>{
  buildSidebar();
  setupScroll();
  window.addEventListener('resize', ()=>{
    resizeCanvases(); requestAnimationFrame(render);
  });
  await preloadImages();
  if (CHARTS.length) selectChart(CHARTS[0].id);
});
</script>
</body>
</html>
"""


def generate_html(charts: list, img_dir_rel: str = 'images') -> str:
    charts_json = json.dumps(charts, ensure_ascii=False, separators=(',', ':'))
    html = HTML_TEMPLATE
    html = html.replace('/*CHARTS_DATA*/', charts_json)
    html = html.replace('/*IMG_DIR*/', img_dir_rel)
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PIU limb diff viewer generator')
    parser.add_argument('--n_charts', type=int, default=40)
    parser.add_argument('--mode', default='all', choices=['S', 'D', 'all'])
    parser.add_argument('--song', default=None)
    parser.add_argument('--out', default=OUT_DIR)
    parser.add_argument('--no_open', action='store_true', help='No abrir browser al terminar')
    args = parser.parse_args()

    print(f'Indexando vis-ss ({VIS_DIR})...')
    vis_idx = load_vis_index(VIS_DIR)
    print(f'  {len(vis_idx)} charts')

    modes = ['S', 'D'] if args.mode == 'all' else [args.mode]
    all_charts = []
    for m in modes:
        batch = build_chart_data(PROC_DIR, vis_idx, args.n_charts, m, args.song)
        print(f'  {m}: {len(batch)} charts')
        all_charts += batch

    if not all_charts:
        print('ERROR: sin charts. Verifica VIS_DIR y PROC_DIR.')
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    img_dst = os.path.join(args.out, 'images')
    if not os.path.exists(img_dst):
        print(f'Copiando imágenes → {img_dst}')
        shutil.copytree(IMG_SRC, img_dst)

    html = generate_html(all_charts, 'images')
    out_path = os.path.join(args.out, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) // 1024
    print(f'\n✓ {out_path}  ({len(all_charts)} charts · {size_kb} KB)')
    if not args.no_open:
        os.system(f'open "{out_path}"')


if __name__ == '__main__':
    main()
