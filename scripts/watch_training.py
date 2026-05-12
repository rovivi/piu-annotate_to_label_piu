#!/usr/bin/env python3
"""
🎯 MLX Training Monitor — Rich-powered animated dashboard.

Usage:
    python scripts/watch_training.py              # auto-detect, live mode
    python scripts/watch_training.py --log FILE   # specific log
    python scripts/watch_training.py --once       # one-shot snapshot
    python scripts/watch_training.py --poll 5     # custom interval
"""
from __future__ import annotations
import os
import sys
import re
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text
from rich.align import Align

console = Console()


def parse_epoch_line(line: str) -> dict | None:
    m = re.search(
        r'Epoch (\d+)/(\d+): train_loss=([\d.]+), val_loss=([\d.]+), val_acc=([\d.]+)%, '
        r'L=([\d.]+)%\(n=\d+\) R=([\d.]+)%\(n=\d+\) E=([\d.]+)%\(n=\d+\), lr=([\d.e+-]+)',
        line
    )
    if m:
        return {
            'epoch': int(m.group(1)),
            'total': int(m.group(2)),
            'train_loss': float(m.group(3)),
            'val_loss': float(m.group(4)),
            'val_acc': float(m.group(5)),
            'L': float(m.group(6)),
            'R': float(m.group(7)),
            'E': float(m.group(8)),
            'lr': m.group(9),
        }
    return None


def parse_step_line(line: str) -> dict | None:
    m = re.search(r'step (\d+), loss=([\d.]+), lr=([\d.e+-]+)', line)
    if m:
        return {'step': int(m.group(1)), 'loss': float(m.group(2)), 'lr': m.group(3)}
    return None


def parse_best_line(line: str) -> dict | None:
    m = re.search(r'New best saved: val_acc=([\d.]+)%', line)
    if m:
        return {'best_acc': float(m.group(1))}
    return None


def find_latest_log() -> str | None:
    logs = sorted(Path('.').glob('out_mlx*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(logs[0]) if logs else None


def make_epoch_table(epochs: list[dict], best_acc: float | None) -> Table:
    table = Table(show_header=True, header_style="bold magenta", expand=True, box=None)
    table.add_column("Ep", justify="right", width=5)
    table.add_column("acc", justify="right", width=7)
    table.add_column("Δ", justify="right", width=5)
    table.add_column("tLoss", justify="right", width=7)
    table.add_column("vLoss", justify="right", width=7)
    table.add_column("L", justify="right", width=5)
    table.add_column("R", justify="right", width=5)
    table.add_column("lr", justify="right", width=9)

    for idx, e in enumerate(epochs[-10:]):
        if idx > 0:
            prev = epochs[-14:][idx - 1]['val_acc']
            delta = e['val_acc'] - prev
            delta_str = f"[green]+{delta:.1f}[/green]" if delta > 0 else f"[red]{delta:.1f}[/red]" if delta < 0 else "[dim]—[/dim]"
        else:
            delta_str = "[dim]—[/dim]"

        # Color by accuracy
        if e['val_acc'] >= 91.0:
            acc_style = "bold white on green"
        elif e['val_acc'] >= 88.0:
            acc_style = "bold green"
        elif e['val_acc'] >= 85.0:
            acc_style = "bold yellow"
        else:
            acc_style = "bold red"

        marker = "▶ " if best_acc and abs(e['val_acc'] - best_acc) < 0.05 else "  "
        table.add_row(
            f"{marker}{e['epoch']}/{e['total']}",
            f"[{acc_style}]{e['val_acc']:.1f}[/{acc_style}]",
            delta_str,
            f"{e['train_loss']:.3f}",
            f"{e['val_loss']:.3f}",
            f"{e['L']:.0f}",
            f"{e['R']:.0f}",
            f"[dim]{e['lr']}[/dim]",
        )
    return table


def make_sparkline(values: list[float], width: int = 40, lo: float = 75.0, hi: float = 95.0) -> str:
    if not values:
        return " " * width
    bars = "▁▂▃▄▅▆▇█"
    out = []
    for v in values[-width:]:
        idx = int((v - lo) / (hi - lo) * (len(bars) - 1))
        idx = max(0, min(len(bars) - 1, idx))
        out.append(bars[idx])
    return "".join(out)


def build_layout(epochs: list[dict], last_step: dict | None, best_acc: float | None,
                 log_path: str, start_time: float) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="hero", size=3),
        Layout(name="sparkline", size=2),
        Layout(name="table", ratio=3),
        Layout(name="footer", size=5),
    )

    now = datetime.now().strftime('%H:%M:%S')
    header_text = f"[bold cyan]🧠 MLX TRAINING DASHBOARD[/bold cyan]    [dim]│[/dim]    [cyan]{now}[/cyan]    [dim]│[/dim]    [yellow blink]⚡ LIVE[/yellow blink]"
    layout["header"].update(Panel(Align.center(header_text), border_style="cyan", padding=(0, 0)))

    # Hero stats
    hero_parts = []
    if best_acc:
        trophy = "🏆" if best_acc >= 91 else "🥈" if best_acc >= 88 else "🥉"
        hero_parts.append(f"[bold white on blue] {trophy} BEST: {best_acc:.1f}% [/bold white on blue]")
    if last_step:
        hero_parts.append(f"📍 step [yellow]{last_step['step']}[/yellow]  loss=[cyan]{last_step['loss']:.4f}[/cyan]  lr=[magenta]{last_step['lr']}[/magenta]")
    hero_text = "   ".join(hero_parts)
    layout["hero"].update(Panel(Align.center(hero_text), border_style="blue", padding=(0, 0)))

    # Sparkline
    if epochs:
        accs = [e['val_acc'] for e in epochs]
        sl = make_sparkline(accs, width=50)
        spark_text = f"[green]{sl}[/green]  [dim]▁75 ▃85 ▇95[/dim]"
        layout["sparkline"].update(Panel(Align.center(spark_text), border_style="green", padding=(0, 0)))

    # Epoch table
    table = make_epoch_table(epochs, best_acc)
    layout["table"].update(Panel(table, border_style="magenta", title="[bold]Epoch History[/bold]", padding=(0, 0)))

    # Footer: progress + stats
    footer_parts = []
    if epochs:
        latest = epochs[-1]
        progress = latest['epoch'] / latest['total']

        # Progress bar
        bar = Progress(
            TextColumn("[bold blue]Epoch Progress"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green", pulse_style="yellow"),
            TextColumn("[bold]{task.percentage:.0f}%"),
            expand=False,
        )
        bar_task = bar.add_task("progress", total=latest['total'], completed=latest['epoch'])
        footer_parts.append(bar)

        # ETA
        elapsed = time.time() - start_time
        if progress > 0:
            total_est = elapsed / progress
            remaining = total_est - elapsed
            eta_mins = int(remaining / 60)
            eta_hrs = eta_mins // 60
            eta_mins %= 60
            eta_str = f"{eta_hrs}h {eta_mins}m" if eta_hrs > 0 else f"{eta_mins}m"
        else:
            eta_str = "???"
        footer_parts.append(Text(f"⏱  ETA remaining: {eta_str}    ({latest['epoch']}/{latest['total']} epochs)"))

        # Gaps
        if best_acc:
            gap_v8 = 91.4 - best_acc
            gap_96 = 96.0 - best_acc
            if gap_v8 > 0:
                footer_parts.append(Text(f"📊 Gap to v8 baseline (91.4%): {gap_v8:.1f}pp", style="yellow"))
            else:
                footer_parts.append(Text(f"🎉 SURPASSED v8 baseline! +{abs(gap_v8):.1f}pp", style="bold green"))
            footer_parts.append(Text(f"🎯 Gap to target (96.0%): {gap_96:.1f}pp", style="cyan"))

    from rich.columns import Columns
    footer_content = Columns(footer_parts, equal=False, expand=True)
    layout["footer"].update(Panel(footer_content, border_style="yellow", padding=(0, 0)))

    return layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--poll', type=int, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    if not log_path or not os.path.exists(log_path):
        console.print("[red]❌ No log file found. Use --log to specify one.[/red]")
        return

    start_time = time.time()

    def read_data():
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
        epochs = []
        last_step = None
        best_acc = None
        for line in content.split('\n'):
            ep = parse_epoch_line(line)
            if ep:
                epochs.append(ep)
            st = parse_step_line(line)
            if st:
                last_step = st
            bst = parse_best_line(line)
            if bst:
                best_acc = bst['best_acc']
        return epochs, last_step, best_acc

    if args.once:
        epochs, last_step, best_acc = read_data()
        layout = build_layout(epochs, last_step, best_acc, log_path, start_time)
        console.print(layout)
        return

    console.print("[dim]Starting live monitor... Press Ctrl+C to exit[/dim]")
    time.sleep(0.5)

    with Live(refresh_per_second=4, screen=True) as live:
        try:
            while True:
                epochs, last_step, best_acc = read_data()
                layout = build_layout(epochs, last_step, best_acc, log_path, start_time)
                live.update(layout)
                time.sleep(args.poll)
        except KeyboardInterrupt:
            pass

    console.print("[dim]👋 Monitor stopped.[/dim]")


if __name__ == '__main__':
    main()
