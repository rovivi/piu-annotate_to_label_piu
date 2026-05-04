#!/usr/bin/env python3
"""
Phase A.1: Diagnosticar errores de bracket pie derecho.

Para cada bracket en el dataset de ground truth (visss-120524-eaware):
- Clasifica por tipo: LL, RR, LR
- Compara predicción vs ground truth
- Genera tabla de accuracy y errores para inspección.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from loguru import logger

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats import notelines


def classify_bracket(row_idx: int, line: str, limb_annot: str, pred_annot: str,
                     pred_coords: list, arrow_pos_to_limb_idx: dict,
                     num_downpress: int) -> tuple[str | None, bool]:
    if num_downpress < 2:
        return None, False
    if not notelines.line_is_bracketable(line):
        return None, False

    ll_count = sum(1 for c in limb_annot if c == 'l')
    rr_count = sum(1 for c in limb_annot if c == 'r')
    lr_count = sum(1 for c in limb_annot if c in 'eh')

    if ll_count >= 2 and rr_count == 0:
        gt_type = 'LL'
    elif rr_count >= 2 and ll_count == 0:
        gt_type = 'RR'
    elif ll_count >= 1 and rr_count >= 1:
        gt_type = 'LR'
    else:
        return None, False

    pred_ll_count = sum(1 for c in pred_annot if c == 'l')
    pred_rr_count = sum(1 for c in pred_annot if c == 'r')
    pred_lr_count = sum(1 for c in pred_annot if c in 'eh')

    if pred_ll_count >= 2 and pred_rr_count == 0:
        pred_type = 'LL'
    elif pred_rr_count >= 2 and pred_ll_count == 0:
        pred_type = 'RR'
    elif pred_ll_count >= 1 and pred_rr_count >= 1:
        pred_type = 'LR'
    else:
        pred_type = None

    correct = (gt_type == pred_type)
    return gt_type, correct


def get_prev_orientation(cs: ChartStruct, row_idx: int, pred_coords: list,
                         limb_annot: str, pred_limb_idxs: list,
                         window: int = 8) -> str | None:
    for look_idx in range(row_idx - 1, max(0, row_idx - window) - 1, -1):
        try:
            pc_idx = pred_limb_idxs.index(look_idx)
        except ValueError:
            continue
        if pc_idx < len(limb_annot):
            a = limb_annot[pc_idx]
            if a in 'lr':
                return a
    return None


def run_diagnosis(gt_folder: str, pred_folder: str, output_dir: str):
    gt_files = list(Path(gt_folder).glob('*.csv'))
    logger.info(f"Found {len(gt_files)} ground truth files")

    stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'fp': defaultdict(int)})

    all_errors = []
    rr_errors = []

    for f in gt_files:
        try:
            cs = ChartStruct.from_file(str(f))
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")
            continue

        pred_file = Path(pred_folder) / f.with_suffix('.json').name
        if not pred_file.exists():
            logger.debug(f"No prediction file for {f.name}, skipping")
            continue

        try:
            from piu_annotate.ml.predictor import Predictor
            pred = Predictor()
            pred.load_from_json(str(pred_file))
            pred_limb_annot = pred.predict_all()
        except Exception as e:
            logger.warning(f"Error predicting {f.name}: {e}")
            continue

        pred_coords = cs.get_prediction_coordinates()
        lines = cs.get_lines_with_active_holds()
        gt_limb_annots = cs.df['Limb annotation'].tolist()

        pred_limb_idxs = [pc.row_idx for pc in pred_coords]

        for row_idx, (line, gt_annot, pred_annot) in enumerate(zip(lines, gt_limb_annots, pred_limb_annot)):
            if not gt_annot or not pred_annot:
                continue

            num_downpress = line.count('1') + line.count('2')
            result = classify_bracket(row_idx, line, gt_annot, pred_annot,
                                     pred_coords, {}, num_downpress)
            if result[0] is None:
                continue

            gt_type, correct = result
            stats[gt_type]['total'] += 1
            if correct:
                stats[gt_type]['correct'] += 1
            else:
                prev_orient = get_prev_orientation(cs, row_idx, pred_coords,
                                                  gt_annot, pred_limb_idxs)
                error = {
                    'chart': f.name,
                    'row': row_idx,
                    'line': line,
                    'gt_annot': gt_annot,
                    'pred_annot': pred_annot,
                    'gt_type': gt_type,
                    'prev_orientation': prev_orient,
                }
                all_errors.append(error)
                if gt_type == 'RR':
                    rr_errors.append(error)

    print("\n" + "="*70)
    print("BRACKET DIAGNOSTIC RESULTS")
    print("="*70)
    print(f"{'Bracket type':<12} {'N':>8} {'Acc':>8} {'FP→LL':>8} {'FP→RR':>8} {'FP→LR':>8}")
    print("-"*70)

    for btype in ['LL', 'RR', 'LR']:
        s = stats[btype]
        if s['total'] == 0:
            continue
        acc = s['correct'] / s['total'] * 100
        fp_ll = s['fp']['LL'] / s['total'] * 100 if s['fp']['LL'] else 0
        fp_rr = s['fp']['RR'] / s['total'] * 100 if s['fp']['RR'] else 0
        fp_lr = s['fp']['LR'] / s['total'] * 100 if s['fp']['LR'] else 0
        print(f"{btype:<12} {s['total']:>8} {acc:>7.1f}% {fp_ll:>7.1f}% {fp_rr:>7.1f}% {fp_lr:>7.1f}%")

    rr_df = pd.DataFrame(rr_errors[:100])
    if len(rr_df) > 0:
        rr_df.to_csv(f"{output_dir}/rr_bracket_errors.csv", index=False)
        logger.success(f"Saved {len(rr_errors)} RR bracket errors to {output_dir}/rr_bracket_errors.csv")

    logger.info("Diagnosis complete")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str,
                        default='artifacts/manual-chartstructs/visss-120524-eaware/')
    parser.add_argument('--pred_folder', type=str,
                        default='artifacts/processed_db/')
    parser.add_argument('--output_dir', type=str,
                        default='artifacts/debug/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_diagnosis(args.gt_folder, args.pred_folder, args.output_dir)