#!/usr/bin/env python3
"""Pre-featurize chartstructs and cache as .npz per chart. Run once."""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.mirror import mirror_chartstruct
from piu_annotate.ml.featurizers import ChartStructFeaturizer


def featurize(cs):
    ft = ChartStructFeaturizer(cs)
    x_raw = np.stack(ft.pt_array)
    cmf = np.tile(ft.chart_metadata_features, (len(x_raw), 1))
    x = np.concatenate([x_raw, cmf], axis=1).astype(np.float32)
    y = ft.get_labels_from_limb_col('Limb annotation').astype(np.int8)
    return x, y


def main(folder: str, out_dir: str, sd: str):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(Path(folder).glob('*.csv'))
    skipped = 0
    existing = 0
    for f in tqdm(files, desc=f'Caching {sd}'):
        out_path = os.path.join(out_dir, f'{f.stem}.npz')
        if os.path.exists(out_path):
            existing += 1
            continue
        try:
            cs = ChartStruct.from_file(str(f))
            if cs.singles_or_doubles() != sd:
                continue
            x, y = featurize(cs)
            xm, ym = featurize(mirror_chartstruct(cs))
            np.savez(out_path, x=x, y=y, x_mirror=xm, y_mirror=ym)
        except Exception as e:
            skipped += 1
    print(f'Done. Skipped {skipped} charts. Existing {existing} charts.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sd', required=True, choices=['singles', 'doubles'])
    args = ap.parse_args()
    main(args.folder, args.out_dir, args.sd)
