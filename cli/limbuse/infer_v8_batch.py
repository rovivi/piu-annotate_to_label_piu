#!/usr/bin/env python3
"""
Batch inference with v8/v9 MLX model on all SSC sim files.
Supports Test-Time Augmentation (TTA) for improved accuracy.

Usage:
    python cli/limbuse/infer_v8_batch.py \
        --simfiles_dir /path/to/piu_sim_files \
        --out_dir comparations/generated \
        --model_dir artifacts/models/visss-mlx-v9 \
        --tta

Skips: UCS, nonstandard, coop, hidden, quest charts.
Outputs one JSON per chart in vis-ss format: [arrows, holds, metadata]
Also writes comparations/index.json for the comparison viewer.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import mlx.nn as nn

from piu_annotate.formats.sscfile import SongSSC, StepchartSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate.formats.mirror import mirror_chartstruct
from piu_annotate.crawl import crawl_sscs
from piu_annotate.ml.featurizers import ChartStructFeaturizer
from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer
from piu_annotate.formats.notelines import fix_impossible_predictions

MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 256
SINGLES_INPUT_DIM = 28  # 24 arrow + 4 prev_limb one-hot
DOUBLES_INPUT_DIM = 33  # 29 arrow + 4 prev_limb one-hot


def load_v8_model(model_dir: str, sd: str) -> LimbSequenceTransformer:
    config_path = os.path.join(model_dir, 'train_config.json')
    with open(config_path) as f:
        cfg = json.load(f)
    input_dim = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM
    model = LimbSequenceTransformer(
        input_dim=input_dim,
        d_model=cfg.get('d_model', 384),
        n_heads=cfg.get('n_heads', 8),
        n_layers=cfg.get('n_layers', 8),
        ffn_dim=cfg.get('ffn_dim', 1536),
    )
    weights_path = os.path.join(model_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
    model.load_weights(weights_path)
    model.eval()
    logger.info(f'Loaded v8 model from {weights_path}')
    return model


def _prev_limb_onehot_from_preds(preds: np.ndarray, first_label: int = 3) -> np.ndarray:
    N = len(preds)
    prev = np.zeros((N, 4), dtype=np.float32)
    prev[0, min(first_label, 3)] = 1.0
    for i in range(1, N):
        lbl = min(int(preds[i - 1]), 2)
        prev[i, lbl] = 1.0
    return prev


def make_chunks(x: np.ndarray, max_len: int = MAX_SEQ_LEN, overlap: int = CHUNK_OVERLAP):
    N = len(x)
    if N <= max_len:
        return [(x, slice(0, N))]
    chunks = []
    stride = max_len - overlap
    start = 0
    while start + max_len <= N:
        chunks.append((x[start:start + max_len], slice(start, start + max_len)))
        start += stride
    if start < N:
        chunks.append((x[N - max_len:], slice(N - max_len, N)))
    return chunks


def model_forward(model: LimbSequenceTransformer, x_chunk: np.ndarray) -> np.ndarray:
    """Run a single chunk through the model, returns softmax probs (L, 3)."""
    L = len(x_chunk)
    x_mx = mx.array(x_chunk[None])           # (1, L, D)
    pm = mx.zeros((1, L), dtype=mx.bool_)    # no padding
    logits = model(x_mx, pm)                 # (1, L, 3)
    mx.eval(logits)
    probs = np.array(mx.softmax(logits, axis=-1))[0]  # (L, 3)
    return probs


def predict_sequence(model: LimbSequenceTransformer, x_full: np.ndarray) -> np.ndarray:
    """Predict over a full sequence with overlapping chunks; average overlaps."""
    N = len(x_full)
    prob_acc = np.zeros((N, 3), dtype=np.float64)
    weight_acc = np.zeros(N, dtype=np.float64)
    for chunk, slc in make_chunks(x_full):
        probs = model_forward(model, chunk)
        chunk_len = slc.stop - slc.start
        # linear weight ramp to down-weight chunk boundaries
        w = np.ones(chunk_len, dtype=np.float64)
        if chunk_len > 2 * CHUNK_OVERLAP:
            ramp = np.linspace(0.3, 1.0, CHUNK_OVERLAP)
            w[:CHUNK_OVERLAP] = ramp
            w[-CHUNK_OVERLAP:] = ramp[::-1]
        prob_acc[slc] += probs * w[:, None]
        weight_acc[slc] += w
    weight_acc = np.maximum(weight_acc, 1e-8)
    return np.argmax(prob_acc / weight_acc[:, None], axis=-1).astype(np.int32)


def predict_sequence_logits(model: LimbSequenceTransformer, x_full: np.ndarray) -> np.ndarray:
    """Predict over a full sequence, returning averaged logits (N, 3)."""
    N = len(x_full)
    logits_acc = np.zeros((N, 3), dtype=np.float64)
    weight_acc = np.zeros(N, dtype=np.float64)
    for chunk, slc in make_chunks(x_full):
        L = len(chunk)
        x_mx = mx.array(chunk[None])
        pm = mx.zeros((1, L), dtype=mx.bool_)
        logits = model(x_mx, pm)
        mx.eval(logits)
        logits_np = np.array(logits)[0]
        chunk_len = slc.stop - slc.start
        w = np.ones(chunk_len, dtype=np.float64)
        if chunk_len > 2 * CHUNK_OVERLAP:
            ramp = np.linspace(0.3, 1.0, CHUNK_OVERLAP)
            w[:CHUNK_OVERLAP] = ramp
            w[-CHUNK_OVERLAP:] = ramp[::-1]
        logits_acc[slc] += logits_np * w[:, None]
        weight_acc[slc] += w
    weight_acc = np.maximum(weight_acc, 1e-8)
    return logits_acc / weight_acc[:, None]


def ar_infer(model: LimbSequenceTransformer, x_base: np.ndarray) -> np.ndarray:
    """2-pass autoregressive inference.
    Pass 1: all prev_limb = START token.
    Pass 2: prev_limb from pass-1 predictions.
    """
    N = len(x_base)
    # Pass 1
    prev1 = np.zeros((N, 4), dtype=np.float32)
    prev1[:, 3] = 0.0
    prev1[0, 3] = 1.0  # START token
    x1 = np.concatenate([x_base, prev1], axis=1)
    preds1 = predict_sequence(model, x1)

    # Pass 2: prev_limb from preds1
    prev2 = _prev_limb_onehot_from_preds(preds1, first_label=3)
    x2 = np.concatenate([x_base, prev2], axis=1)
    preds2 = predict_sequence(model, x2)
    return preds2


def ar_infer_logits(model: LimbSequenceTransformer, x_base: np.ndarray) -> np.ndarray:
    """2-pass autoregressive inference returning logits."""
    N = len(x_base)
    prev1 = np.zeros((N, 4), dtype=np.float32)
    prev1[:, 3] = 0.0
    prev1[0, 3] = 1.0
    x1 = np.concatenate([x_base, prev1], axis=1)
    logits1 = predict_sequence_logits(model, x1)
    preds1 = np.argmax(logits1, axis=-1)

    prev2 = _prev_limb_onehot_from_preds(preds1, first_label=3)
    x2 = np.concatenate([x_base, prev2], axis=1)
    logits2 = predict_sequence_logits(model, x2)
    return logits2


def flip_lr_logits(logits: np.ndarray) -> np.ndarray:
    """Swap L and R logits: logits[:, [1, 0, 2]]"""
    flipped = logits.copy()
    flipped[:, [0, 1]] = logits[:, [1, 0]]
    return flipped


def infer_chart(cs: ChartStruct, model: LimbSequenceTransformer, use_tta: bool = False) -> ChartStruct | None:
    """Run v8/v9 model on a ChartStruct and write limb annotations. Returns None on failure."""
    try:
        sd = cs.singles_or_doubles()
        fcs = ChartStructFeaturizer(cs)
        x_base = fcs.get_raw_features().astype(np.float32)  # N × D
        if len(x_base) == 0:
            return None

        if use_tta:
            # Pass 1: original
            logits_orig = ar_infer_logits(model, x_base)

            # Pass 2: mirrored chart
            cs_mirror = mirror_chartstruct(cs)
            fcs_mirror = ChartStructFeaturizer(cs_mirror)
            x_base_mirror = fcs_mirror.get_raw_features().astype(np.float32)
            logits_mirror = ar_infer_logits(model, x_base_mirror)

            # Flip mirrored logits back
            logits_mirror_flipped = flip_lr_logits(logits_mirror)

            # Average
            logits_final = (logits_orig + logits_mirror_flipped) * 0.5
            preds = np.argmax(logits_final, axis=-1).astype(np.int32)
        else:
            preds = ar_infer(model, x_base)

        # Hard-fix impossible same-foot bracket assignments
        preds = fix_impossible_predictions(preds, x_base[:, 0], x_base[:, 6])

        pred_coords = cs.get_prediction_coordinates()
        pred_limb_strs = [INT_TO_LIMB[int(p)] for p in preds]
        cs.add_limb_annotations(pred_coords, pred_limb_strs, 'Limb annotation')
        cs.metadata['Manual limb annotation'] = False
        cs.metadata['generated_by'] = 'v9_mlx' + ('_tta' if use_tta else '')
        return cs
    except Exception as e:
        logger.warning(f'Failed to infer: {e}')
        return None


INT_TO_LIMB = {0: 'l', 1: 'r', 2: 'e'}


def cs_to_json(cs: ChartStruct) -> list:
    """Convert ChartStruct with limb annotations to vis-ss JSON format."""
    cjs = ChartJsStruct.from_chartstruct(cs)
    arrows = [aa.to_tuple() for aa in cjs.arrow_arts]
    holds = [ha.to_tuple() for ha in cjs.hold_arts]
    return [arrows, holds, cjs.metadata]


def compute_accuracy_vs_origin(gen_json: list, orig_json: list) -> float | None:
    """Compare generated vs origin limb annotations. Returns fraction correct."""
    gen_arrows = gen_json[0]
    orig_arrows = orig_json[0]
    if len(gen_arrows) != len(orig_arrows):
        return None
    # Match by (panel, time) and compare limb
    orig_map = {(round(a[1], 3), int(a[0])): a[2] for a in orig_arrows}
    correct = total = 0
    for a in gen_arrows:
        key = (round(float(a[1]), 3), int(a[0]))
        if key in orig_map:
            limb_gen = a[2].lower()
            limb_orig = orig_map[key].lower()
            if limb_orig in ('l', 'r'):  # only count definite labels
                total += 1
                if limb_gen == limb_orig:
                    correct += 1
    if total == 0:
        return None
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--simfiles_dir', default='/Users/rodrigo/dev/piu/piu_sim_files')
    ap.add_argument('--out_dir', default='comparations/generated')
    ap.add_argument('--model_dir', default='artifacts/models/visss-mlx-v9')
    ap.add_argument('--origin_viss_dir', default='comparations/origin_viss')
    ap.add_argument('--viss_src', default='/Users/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524')
    ap.add_argument('--sd', default='singles', choices=['singles', 'doubles', 'both'])
    ap.add_argument('--limit', type=int, default=None, help='Limit charts for testing')
    ap.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.origin_viss_dir, exist_ok=True)

    sds = ['singles', 'doubles'] if args.sd == 'both' else [args.sd]

    # Build vis-ss index: shortname → filepath
    logger.info('Indexing vis-ss JSONs...')
    viss_index: dict[str, str] = {}
    if os.path.isdir(args.viss_src):
        for fn in os.listdir(args.viss_src):
            if fn.endswith('.json'):
                shortname = fn[:-5]
                viss_index[shortname] = os.path.join(args.viss_src, fn)
    logger.info(f'Found {len(viss_index)} vis-ss charts')

    # Crawl sim files
    logger.info(f'Crawling {args.simfiles_dir}...')
    song_sscs = crawl_sscs(args.simfiles_dir, skip_packs=[])
    stepcharts: list[StepchartSSC] = []
    for song in song_sscs:
        stepcharts += song.stepcharts

    # Filter out nonstandard (UCS, coop, quest, hidden, etc.)
    standard = [sc for sc in stepcharts if not sc.is_nonstandard()]
    logger.info(f'Total stepcharts: {len(stepcharts)}, standard: {len(standard)}')

    index_entries = []
    stats = {'ok': 0, 'skipped': 0, 'failed': 0, 'matched_viss': 0}

    for sd in sds:
        sd_charts = [sc for sc in standard if
                     ('double' in sc.data.get('STEPSTYPE', '').lower()) == (sd == 'doubles')]
        if args.limit:
            sd_charts = sd_charts[:args.limit]

        logger.info(f'Processing {len(sd_charts)} {sd} charts...')
        model = load_v8_model(args.model_dir, sd)

        for stepchart in tqdm(sd_charts, desc=f'{sd}'):
            shortname = stepchart.shortname()
            out_path = os.path.join(args.out_dir, f'{shortname}.json')

            # Skip if already done
            if os.path.exists(out_path):
                stats['skipped'] += 1
                # Still add to index
                with open(out_path) as f:
                    gen_json = json.load(f)
                accuracy = None
                has_viss = shortname in viss_index
                if has_viss:
                    with open(viss_index[shortname]) as f:
                        orig_json = json.load(f)
                    accuracy = compute_accuracy_vs_origin(gen_json, orig_json)
                    # Copy origin_viss if not yet copied
                    orig_dest = os.path.join(args.origin_viss_dir, f'{shortname}.json')
                    if not os.path.exists(orig_dest):
                        import shutil
                        shutil.copy2(viss_index[shortname], orig_dest)
                index_entries.append({
                    'shortname': shortname,
                    'sd': sd,
                    'accuracy': round(accuracy * 100, 2) if accuracy is not None else None,
                    'has_viss': has_viss,
                    'pack': stepchart.data.get('pack', ''),
                    'title': stepchart.data.get('TITLE', shortname),
                    'level': stepchart.data.get('METER', '?'),
                })
                continue

            try:
                cs = ChartStruct.from_stepchart_ssc(stepchart)
                cs = infer_chart(cs, model, use_tta=args.tta)
                if cs is None:
                    stats['failed'] += 1
                    continue
                gen_json = cs_to_json(cs)
                with open(out_path, 'w') as f:
                    json.dump(gen_json, f, separators=(',', ':'))
            except Exception as e:
                logger.warning(f'Error on {shortname}: {e}')
                stats['failed'] += 1
                continue

            # Compare vs vis-ss if available
            accuracy = None
            has_viss = shortname in viss_index
            if has_viss:
                try:
                    with open(viss_index[shortname]) as f:
                        orig_json = json.load(f)
                    # Copy origin_viss JSON
                    orig_dest = os.path.join(args.origin_viss_dir, f'{shortname}.json')
                    if not os.path.exists(orig_dest):
                        import shutil
                        shutil.copy2(viss_index[shortname], orig_dest)
                    accuracy = compute_accuracy_vs_origin(gen_json, orig_json)
                    stats['matched_viss'] += 1
                except Exception:
                    pass

            index_entries.append({
                'shortname': shortname,
                'sd': sd,
                'accuracy': round(accuracy * 100, 2) if accuracy is not None else None,
                'has_viss': has_viss,
                'pack': stepchart.data.get('pack', ''),
                'title': stepchart.data.get('TITLE', shortname),
                'level': stepchart.data.get('METER', '?'),
            })
            stats['ok'] += 1

    # Write index
    index_path = os.path.join(os.path.dirname(args.out_dir), 'index.json')
    index_entries.sort(key=lambda x: (x['accuracy'] is not None, x['accuracy'] or 0))
    with open(index_path, 'w') as f:
        json.dump(index_entries, f, indent=2)
    logger.success(f'Wrote index to {index_path}')
    logger.success(f'Stats: {stats}')


if __name__ == '__main__':
    main()
