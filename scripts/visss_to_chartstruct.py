#!/usr/bin/env python3
"""
Convert vis-ss JSON format to ChartStruct CSV format for ML training.

vis-ss format: [taps, holds, metadata]
- taps: [[panel, time, "l"|"r"], ...]
- holds: [[panel, start_time, end_time, "l"|"r"], ...]
- metadata: dict with shortname, STEPSTYPE, etc.

Output: CSV with Beat, Time, Line, Line with active holds, Limb annotation
"""

import json
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

VISS_DIR = "/home/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524"
OUT_DIR = "/home/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524"


def parse_viss_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    taps, holds, metadata = data[0], data[1], data[2]
    return taps, holds, metadata


def get_line_length(stepstype):
    if "pump-double" in stepstype.lower():
        return 10
    return 5


def process_viss_to_csv(filepath, out_dir):
    basename = os.path.basename(filepath)
    shortname = basename.replace(".json", "")

    taps, holds, metadata = parse_viss_json(filepath)
    stepstype = metadata.get("STEPSTYPE", "pump-single")
    line_len = get_line_length(stepstype)

    all_panels = set(t[0] for t in taps) | set(h[0] for h in holds)
    max_panel = max(all_panels) if all_panels else 0

    if stepstype == "pump-halfdouble":
        line_len = max(max_panel + 1, 8)
    elif max_panel >= 8 and "pump-double" not in stepstype.lower():
        line_len = max(max_panel + 1, 5)

    time_to_taps = defaultdict(list)
    for panel, time, limb in taps:
        time_key = round(time, 4)
        time_to_taps[time_key].append((panel, limb))

    holds_by_start = defaultdict(list)
    holds_by_end = defaultdict(list)
    for panel, start, end, limb in holds:
        start_key = round(start, 4)
        end_key = round(end, 4)
        holds_by_start[start_key].append((panel, end_key, limb))
        holds_by_end[end_key].append((panel, start_key, limb))

    all_times = sorted(set(time_to_taps.keys()) | set(holds_by_start.keys()) | set(holds_by_end.keys()))
    if not all_times:
        return None

    rows = []
    active_holds = {}

    for time_key in all_times:
        line = ["0"] * line_len
        line_ah = ["0"] * line_len
        limb_annot = []

        if time_key in holds_by_end:
            for panel, start_key, limb in holds_by_end[time_key]:
                if panel < line_len:
                    line[panel] = "3"
                    line_ah[panel] = "3"
                if panel in active_holds:
                    del active_holds[panel]

        if time_key in holds_by_start:
            for panel, end_key, limb in holds_by_start[time_key]:
                if panel < line_len:
                    line[panel] = "2"
                    line_ah[panel] = "2"
                    limb_annot.append(limb)
                active_holds[panel] = (panel, end_key, limb)

        if time_key in time_to_taps:
            for panel, limb in time_to_taps[time_key]:
                if panel < line_len:
                    line[panel] = "1"
                    line_ah[panel] = "1"
                    limb_annot.append(limb)

        for panel, (p2, end_key, limb) in active_holds.items():
            if panel < line_len:
                line[panel] = "0"
                line_ah[panel] = "4"

        beat_estimate = len(rows)
        row = {
            "Beat": beat_estimate,
            "Time": time_key,
            "Line": "`" + "".join(line),
            "Line with active holds": "`" + "".join(line_ah),
            "Limb annotation": "".join(limb_annot),
        }
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"{shortname}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(VISS_DIR) if f.endswith(".json")]
    print(f"Found {len(files)} vis-ss JSON files")

    successes = 0
    failures = 0
    for fname in tqdm(files):
        filepath = os.path.join(VISS_DIR, fname)
        try:
            result = process_viss_to_csv(filepath, OUT_DIR)
            if result:
                successes += 1
            else:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"Error processing {fname}: {e}")

    print(f"\nDone: {successes} converted, {failures} failed")


if __name__ == "__main__":
    main()