#!/usr/bin/env python3
"""Real-time training dashboard for PIU limb models.

Tails a loguru training log and renders a live terminal view: epoch/step
progress bars, loss sparkline, val-acc history chart, per-class accuracy,
AR accuracy gap, GPU memory, and ETA.

Usage:
    python scripts/train_dashboard.py                        # auto-detect newest .log
    python scripts/train_dashboard.py --log logs/train.log
    python scripts/train_dashboard.py --log logs/train.log --refresh 1.0
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# ── ANSI ─────────────────────────────────────────────────────────────────────

R  = '\033[0m'
B  = '\033[1m'
DIM = '\033[2m'
RED = '\033[31m'
GRN = '\033[32m'
YLW = '\033[33m'
CYN = '\033[36m'
WHT = '\033[97m'

_ANSI_RE = re.compile(r'\033\[[^m]*m')

def vlen(s: str) -> int:
    return len(_ANSI_RE.sub('', s))


# ── State ─────────────────────────────────────────────────────────────────────

class State:
    def __init__(self):
        self.epoch          = 0
        self.total_epochs   = 0
        self.step           = 0
        self.steps_per_epoch = 0
        self.loss           = None
        self.lr             = None
        self.train_loss     = None
        self.val_loss       = None
        self.val_acc        = None
        self.ar_acc         = None
        self.best_acc       = 0.0
        self.best_epoch     = 0
        self.no_improve     = 0
        self.patience       = None
        self.per_class      = {}        # {0: (acc, n), 1: (acc, n), 2: (acc, n)}
        self.status         = 'waiting'
        self.loss_hist      = deque(maxlen=60)
        self.val_hist       = []        # [(epoch, val_acc)]
        self.start_time     = None
        self.epoch_times    = []
        self._epoch_start   = None
        self.backend        = '?'
        self.sd             = '?'
        self.device         = '?'
        self.n_params       = None
        self.train_files    = None
        self.val_chunks     = None
        self.gpu_info       = ''
        self.log_path       = ''


# ── Parsers ───────────────────────────────────────────────────────────────────

_LOG_LINE   = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \| \w+\s+\|[^-]+-\s*(.*)', re.S)
_EPOCH_HDR  = re.compile(r'=== (?:\[.*?\] )?Epoch (\d+)/(\d+) ===')
_STEP       = re.compile(r'step (\d+), loss=([\d.]+), lr=([\de.+\-]+)')
_EPO_FULL   = re.compile(
    r'Epoch (\d+)(?:/(\d+))?: train_loss=([\d.]+).*?val_loss=([\d.]+).*?val_acc=([\d.]+)%'
    r'.*?L=([\d.]+)%\(n=(\d+)\).*?R=([\d.]+)%\(n=(\d+)\).*?E=([\d.]+)%\(n=(\d+)\)'
)
_EPO_SIMPLE = re.compile(r'Epoch (\d+)(?:/(\d+))?: train_loss=([\d.]+)\s+val_acc=([\d.]+)%')
_AR         = re.compile(r'AR val_acc=([\d.]+)%.*?oracle gap=([\d.]+)pp')
_BEST       = re.compile(r'(?:New best saved|New best).*?val_acc=([\d.]+)%')
_NOIMPROVE  = re.compile(r'No improvement (\d+)/(\d+)')
_STEPS_EPO  = re.compile(r'~?(\d+)\s*steps?/epoch')
_PARAMS     = re.compile(r'(?:Model|refine params):\s*([\d,]+)\s*parameters?')
_DEVICE     = re.compile(r'Device:\s*(\S+)')
_FILES      = re.compile(r'[Tt]rain\s*files?:?\s*(\d+).*?[Vv]al\s*(?:chunks?|files?):\s*(\d+)')
_BACKEND    = re.compile(r'\[(mlx|torch)\]')


def parse_line(line: str, st: State):
    m = _LOG_LINE.match(line)
    if not m:
        return
    msg = m.group(1).strip()
    st.last_update = time.time()

    bm = _BACKEND.search(msg)
    if bm:
        st.backend = bm.group(1)

    # Epoch header
    eh = _EPOCH_HDR.search(msg)
    if eh:
        new_ep = int(eh.group(1))
        st.total_epochs = int(eh.group(2))
        if new_ep != st.epoch:
            st._epoch_start = time.time()
            st.epoch = new_ep
        if st.start_time is None:
            st.start_time = time.time()
        st.status = 'training'
        return

    # Step loss/lr
    sm = _STEP.search(msg)
    if sm:
        st.step = int(sm.group(1))
        st.loss = float(sm.group(2))
        st.lr   = float(sm.group(3))
        st.loss_hist.append(st.loss)
        if st.start_time is None:
            st.start_time = time.time()
        return

    # Steps-per-epoch hint
    spe = _STEPS_EPO.search(msg)
    if spe:
        st.steps_per_epoch = int(spe.group(1))

    # AR accuracy
    ar = _AR.search(msg)
    if ar:
        st.ar_acc = float(ar.group(1))
        return

    # Full epoch result (train_torch.py)
    ef = _EPO_FULL.search(msg)
    if ef:
        ep = int(ef.group(1))
        if ef.group(2):
            st.total_epochs = int(ef.group(2))
        st.train_loss = float(ef.group(3))
        st.val_loss   = float(ef.group(4))
        st.val_acc    = float(ef.group(5))
        st.per_class  = {
            0: (float(ef.group(6)),  int(ef.group(7))),
            1: (float(ef.group(8)),  int(ef.group(9))),
            2: (float(ef.group(10)), int(ef.group(11))),
        }
        st.val_hist.append((ep, st.val_acc))
        if st._epoch_start:
            st.epoch_times.append(time.time() - st._epoch_start)
        return

    # Simple epoch result (train_refine.py)
    es = _EPO_SIMPLE.search(msg)
    if es:
        ep = int(es.group(1))
        if es.group(2):
            st.total_epochs = int(es.group(2))
        st.train_loss = float(es.group(3))
        st.val_acc    = float(es.group(4))
        st.val_hist.append((ep, st.val_acc))
        if st._epoch_start:
            st.epoch_times.append(time.time() - st._epoch_start)
        return

    # New best
    bst = _BEST.search(msg)
    if bst:
        st.best_acc   = float(bst.group(1))
        st.best_epoch = st.epoch
        st.no_improve = 0
        return

    # No improvement
    nim = _NOIMPROVE.search(msg)
    if nim:
        st.no_improve = int(nim.group(1))
        st.patience   = int(nim.group(2))
        return

    if re.search(r'Early stop', msg):
        st.status = 'early_stop'
        return

    if re.search(r'[Tt]raining complete', msg):
        st.status = 'complete'
        return

    pm = _PARAMS.search(msg)
    if pm:
        st.n_params = int(pm.group(1).replace(',', ''))

    dm = _DEVICE.search(msg)
    if dm:
        st.device = dm.group(1)

    fm = _FILES.search(msg)
    if fm:
        st.train_files = int(fm.group(1))
        st.val_chunks  = int(fm.group(2))


# ── GPU polling ───────────────────────────────────────────────────────────────

def _poll_gpu(st: State, interval=5.0):
    while True:
        info = _read_gpu()
        if info:
            st.gpu_info = info
        time.sleep(interval)


def _read_gpu() -> str:
    # rocm-smi
    try:
        r = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--showtemp'],
            capture_output=True, text=True, timeout=4,
        )
        if r.returncode == 0:
            out = r.stdout
            used_m  = re.search(r'VRAM Used.*?:\s*([\d]+)', out)
            total_m = re.search(r'VRAM Total.*?:\s*([\d]+)', out)
            temp_m  = re.search(r'Temperature.*?:\s*([\d.]+)', out)
            parts = []
            if used_m and total_m:
                u = int(used_m.group(1)) // (1024**2)
                t = int(total_m.group(1)) // (1024**2)
                pct = u / max(t, 1) * 100
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                parts.append(f'VRAM [{bar}] {u}/{t} MB ({pct:.0f}%)')
            if temp_m:
                parts.append(f'T:{temp_m.group(1)}°C')
            if parts:
                return '  '.join(parts)
            # Fallback: just show first non-empty line
            lines = [l.strip() for l in out.splitlines() if l.strip() and '===' not in l]
            return ' | '.join(lines[:2]) if lines else ''
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # nvidia-smi fallback
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=4,
        )
        if r.returncode == 0:
            vals = r.stdout.strip().split(', ')
            if len(vals) >= 3:
                u, t, temp = int(vals[0]), int(vals[1]), vals[2]
                pct = u / max(t, 1) * 100
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                return f'VRAM [{bar}] {u}/{t} MB ({pct:.0f}%)  T:{temp}°C'
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return ''


# ── Sparkline ─────────────────────────────────────────────────────────────────

_SPARKS = '▁▂▃▄▅▆▇█'

def sparkline(values, width=56, height=5) -> list[str]:
    if not values:
        return [' (no data yet)']
    vals = list(values)[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1e-9

    rows = []
    for row in range(height - 1, -1, -1):
        lo = mn + rng * row / height
        hi = mn + rng * (row + 1) / height
        line = ''
        for v in vals:
            if v >= hi:
                line += '█'
            elif v >= lo:
                idx = int((v - lo) / (hi - lo) * (len(_SPARKS) - 1))
                line += _SPARKS[idx]
            else:
                line += ' '
        y_label = f'{hi:6.4f} │'
        rows.append(y_label + line)
    return rows


def val_chart(val_hist, width=56, height=5) -> list[str]:
    if not val_hist:
        return [' (no epochs yet)']
    accs = [a for _, a in val_hist]
    mn, mx = min(accs), max(accs) + 0.01
    rng = mx - mn
    rows = []
    display = accs[-(width):]
    for row in range(height - 1, -1, -1):
        lo = mn + rng * row / height
        hi = mn + rng * (row + 1) / height
        line = ''
        for a in display:
            if a >= hi:
                line += '│'
            elif a >= lo:
                line += '●'
            else:
                line += ' '
        y_label = f'{hi:5.1f}% │'
        rows.append(y_label + line)
    return rows


# ── Render ────────────────────────────────────────────────────────────────────

W = 74   # inner width (between the ║ chars)


def _bar(frac: float, w: int = 38, fg=CYN) -> str:
    filled = max(0, min(w, int(frac * w)))
    return fg + '█' * filled + DIM + '░' * (w - filled) + R


def _dur(secs: float) -> str:
    s = int(secs)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f'{h:02d}h{m:02d}m{s:02d}s' if h else f'{m:02d}m{s:02d}s'


def _row(content: str = '') -> str:
    pad = W - vlen(content)
    return '║' + content + ' ' * max(pad, 0) + '║'


def _hdr(text: str) -> str:
    return _row(B + CYN + text + R)


def _sep(c='─') -> str:
    return '╠' + c * W + '╣'


def render(st: State) -> str:
    lines = []

    STATUS_CLR = {
        'waiting': YLW, 'training': GRN, 'complete': CYN, 'early_stop': YLW,
    }
    sc = STATUS_CLR.get(st.status, WHT)

    # Title bar
    title = f'  PIU TRAINING DASHBOARD  │  {B}{st.sd}{R}  │  {st.backend}  │  {st.device}  '
    lines.append('╔' + '═' * W + '╗')
    lines.append(_row(title))

    status_str = f'  Status: {sc}{B}{st.status.upper()}{R}'
    if st.n_params:
        status_str += f'   params: {st.n_params:,}'
    if st.log_path:
        fname = Path(st.log_path).name
        status_str += f'   log: {DIM}{fname}{R}'
    lines.append(_row(status_str))

    # ── Epoch / step progress ─────────────────────────────────────────────────
    lines.append(_sep())
    if st.total_epochs > 0:
        frac = max(0.0, (st.epoch - 1) / st.total_epochs)
        bar  = _bar(frac, 38, CYN)
        lines.append(_row(f'  Epoch {st.epoch}/{st.total_epochs}  [{bar}{R}]  {frac*100:.1f}%'))

    if st.steps_per_epoch > 0 and st.step > 0:
        step_in_ep = st.step % st.steps_per_epoch or st.steps_per_epoch
        sfrac = step_in_ep / st.steps_per_epoch
        sbar  = _bar(sfrac, 38, YLW)
        lines.append(_row(f'  Step  {step_in_ep}/{st.steps_per_epoch}  [{sbar}{R}]  {sfrac*100:.1f}%'))
    elif st.step > 0:
        lines.append(_row(f'  Step: {st.step}'))

    if st.loss is not None:
        lr_str = f'{st.lr:.2e}' if st.lr is not None else '?'
        tl_str = f'   train_loss: {st.train_loss:.4f}' if st.train_loss else ''
        lines.append(_row(f'  Loss: {B}{st.loss:.4f}{R}   LR: {lr_str}{tl_str}'))

    # ── Val accuracy ─────────────────────────────────────────────────────────
    lines.append(_sep())
    lines.append(_hdr('  VAL ACCURACY'))
    if st.val_acc is not None:
        acc_c = GRN if st.val_acc >= 94.0 else (YLW if st.val_acc >= 90.0 else RED)
        acc_str = f'  Current: {acc_c}{B}{st.val_acc:.2f}%{R}'
        if st.best_acc > 0:
            acc_str += f'   Best: {GRN}{B}{st.best_acc:.2f}%{R} (ep {st.best_epoch})'
        if st.patience:
            nim_c = RED if st.no_improve >= st.patience - 1 else YLW
            acc_str += f'   no-impr: {nim_c}{st.no_improve}/{st.patience}{R}'
        lines.append(_row(acc_str))

    if st.ar_acc is not None and st.val_acc is not None:
        gap = st.val_acc - st.ar_acc
        gap_c = GRN if gap < 1.0 else YLW
        lines.append(_row(f'  AR acc: {B}{st.ar_acc:.2f}%{R}   oracle gap: {gap_c}{gap:.2f}pp{R}'))

    if st.per_class:
        la, ln = st.per_class.get(0, (0, 0))
        ra, rn = st.per_class.get(1, (0, 0))
        ea, en = st.per_class.get(2, (0, 0))
        e_c = GRN if ea >= 60 else RED
        lines.append(_row(
            f'  L: {la:.1f}%(n={ln})   R: {ra:.1f}%(n={rn})   E: {e_c}{ea:.1f}%{R}(n={en})'
        ))

    if st.val_loss is not None:
        lines.append(_row(f'  val_loss: {st.val_loss:.4f}'))

    # ── Val acc chart ─────────────────────────────────────────────────────────
    if len(st.val_hist) >= 2:
        lines.append(_sep())
        lines.append(_hdr('  VAL ACC HISTORY (each epoch)'))
        for row in val_chart(st.val_hist, width=W - 10, height=4):
            lines.append(_row('  ' + row))
        ep_labels = '  ' + ' ' * 8
        for i, (ep, _) in enumerate(st.val_hist[-(W - 10):]):
            ep_labels += str(ep % 10)
        lines.append(_row(DIM + ep_labels[:W - 2] + R))

    # ── Loss sparkline ────────────────────────────────────────────────────────
    if len(st.loss_hist) >= 5:
        lines.append(_sep())
        mn_l = min(st.loss_hist)
        mx_l = max(st.loss_hist)
        lines.append(_hdr(f'  STEP LOSS  (last {len(st.loss_hist)}, min={mn_l:.4f} max={mx_l:.4f})'))
        for row in sparkline(st.loss_hist, width=W - 10, height=4):
            lines.append(_row('  ' + row))

    # ── System info ───────────────────────────────────────────────────────────
    lines.append(_sep())
    lines.append(_hdr('  SYSTEM'))
    if st.gpu_info:
        lines.append(_row('  ' + st.gpu_info))
    else:
        lines.append(_row(f'  GPU: {DIM}no rocm-smi / nvidia-smi found{R}'))

    if st.train_files:
        lines.append(_row(f'  Data: {st.train_files} train files  {st.val_chunks} val chunks'))

    if st.start_time:
        elapsed = time.time() - st.start_time
        eta_str = ''
        if st.epoch_times and st.total_epochs > 0 and st.status == 'training':
            avg_ep = sum(st.epoch_times) / len(st.epoch_times)
            remaining = st.total_epochs - st.epoch
            eta_str = f'   ETA: ~{_dur(avg_ep * remaining)}'
        lines.append(_row(f'  Elapsed: {_dur(elapsed)}{eta_str}'))

    lines.append(_row(f'  {DIM}Updated: {datetime.now().strftime("%H:%M:%S")}{R}'))
    lines.append('╚' + '═' * W + '╝')

    return '\n'.join(lines)


# ── Log tailer ────────────────────────────────────────────────────────────────

def tail_file(path: str, st: State):
    with open(path, 'r', errors='replace') as f:
        while True:
            line = f.readline()
            if line:
                parse_line(line, st)
            else:
                time.sleep(0.1)


def find_newest_log() -> str | None:
    root = Path('.')
    candidates = (
        list(root.glob('*.log'))
        + list(root.glob('logs/*.log'))
        + list(root.glob('logs/**/*.log'))
    )
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='PIU real-time training dashboard')
    p.add_argument('--log', default=None, help='Path to loguru training log (auto-detect if omitted)')
    p.add_argument('--refresh', type=float, default=0.5, help='Screen refresh interval (s)')
    p.add_argument('--sd', default=None, help='singles or doubles (for display label)')
    args = p.parse_args()

    log_path = args.log or find_newest_log()
    if not log_path or not os.path.exists(log_path):
        print('No log file found. Pass --log <path> or put logs in logs/.')
        sys.exit(1)

    st = State()
    st.log_path = log_path

    if args.sd:
        st.sd = args.sd
    else:
        lp = log_path.lower()
        st.sd = 'singles' if 'singles' in lp else ('doubles' if 'doubles' in lp else '?')

    threading.Thread(target=tail_file,  args=(log_path, st), daemon=True).start()
    threading.Thread(target=_poll_gpu,  args=(st,),           daemon=True).start()

    try:
        while True:
            out = render(st)
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.write(out)
            sys.stdout.write('\n')
            sys.stdout.flush()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print('\nDashboard stopped.')


if __name__ == '__main__':
    main()
