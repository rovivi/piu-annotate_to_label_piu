"""Numpy-only data plumbing shared by all training/inference backends.

Holds:
  - Chunking helpers (sliding window with overlap)
  - prev_limb one-hot construction
  - Cached-npz loaders
  - Batch builder with length bucketing + impossible-bracket pair indices

Nothing here depends on torch or mlx so it stays cheap to import.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from tqdm import tqdm


MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 256


# Bracketable arrow-position pairs (singles + doubles panels).
# Source of truth lives in tactics.py; mirrored here for fast lookup.
_BRACKETABLE_SET = frozenset(
    (min(a, b), max(a, b)) for a, b in [
        [0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2],
        [4, 5], [3, 6],
        [5, 6], [5, 7], [6, 7], [8, 7], [8, 9], [9, 7],
    ]
)


def find_impossible_pairs(x_np: np.ndarray) -> list[tuple[int, int]]:
    """Return token-index pairs that share a line but would be physically
    impossible as a same-foot bracket."""
    arrow_pos = x_np[:, 0].astype(int)
    num_dp = x_np[:, 6].astype(int)
    n = len(x_np)
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        k = int(num_dp[i])
        if k >= 2:
            grp = list(range(i, min(i + k, n)))
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    p1, p2 = arrow_pos[grp[a]], arrow_pos[grp[b]]
                    if (min(p1, p2), max(p1, p2)) not in _BRACKETABLE_SET:
                        out.append((grp[a], grp[b]))
            i += k
        else:
            i += 1
    return out


def prev_limb_onehot(y: np.ndarray, first_label: int = 3) -> np.ndarray:
    """(N, 4) one-hot of label at i-1. Classes: L=0, R=1, E=2, START=3."""
    n = len(y)
    prev = np.zeros((n, 4), dtype=np.float32)
    prev[0, min(int(first_label), 3)] = 1.0
    for i in range(1, n):
        prev[i, min(int(y[i - 1]), 2)] = 1.0
    return prev


def make_chunks_with_prev(x, y, max_len=MAX_SEQ_LEN, overlap=CHUNK_OVERLAP):
    """Sliding window chunking with prev_limb one-hot appended to features."""
    if len(x) <= max_len:
        prev = prev_limb_onehot(y, first_label=3)
        return [(np.concatenate([x, prev], axis=1), y)]
    chunks = []
    stride = max_len - overlap
    for s in range(0, len(x) - max_len + 1, stride):
        cx = x[s:s + max_len]
        cy = y[s:s + max_len]
        first = 3 if s == 0 else min(int(y[s - 1]), 2)
        prev = prev_limb_onehot(cy, first_label=first)
        chunks.append((np.concatenate([cx, prev], axis=1), cy))
    if (len(x) - max_len) % stride != 0:
        s = len(x) - max_len
        cx = x[s:]
        cy = y[s:]
        first = 3 if s == 0 else min(int(y[s - 1]), 2)
        prev = prev_limb_onehot(cy, first_label=first)
        chunks.append((np.concatenate([cx, prev], axis=1), cy))
    return chunks


def load_train_data_pair(files):
    """Eager-load (x, y, x_mirror, y_mirror) from .npz cache."""
    out = []
    for f in tqdm(files, desc='Loading training data'):
        d = np.load(f)
        out.append((
            d['x'].astype(np.float32),
            d['y'].astype(np.float32),
            d['x_mirror'].astype(np.float32),
            d['y_mirror'].astype(np.float32),
        ))
    return out


def build_both_epoch_chunks(train_data):
    """Both original + mirror versions for every chart, with prev_limb."""
    out = []
    for x, y, xm, ym in train_data:
        out.extend(make_chunks_with_prev(x, y))
        out.extend(make_chunks_with_prev(xm, ym))
    return out


def load_val_chunks(files):
    out = []
    for f in tqdm(files, desc='Loading val data'):
        d = np.load(f)
        x = d['x'].astype(np.float32)
        y = d['y'].astype(np.float32)
        out.extend(make_chunks_with_prev(x, y))
    return out


def build_batches(chunks, batch_size, shuffle=True, seed=0, with_impossible_pairs=False):
    """Yield dict-batches sorted by length to minimise padding waste."""
    sorted_idx = sorted(range(len(chunks)), key=lambda i: chunks[i][0].shape[0])
    batches = [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    for idxs in batches:
        L_max = max(chunks[i][0].shape[0] for i in idxs)
        bx, by, bp, bl = [], [], [], []
        pb, pi_, pj_ = [], [], []
        for bi, ci in enumerate(idxs):
            x, y = chunks[ci]
            L = x.shape[0]
            xp = np.zeros((L_max, x.shape[1]), dtype=np.float32)
            yp = np.full((L_max,), -1, dtype=np.float32)
            xp[:L] = x
            yp[:L] = y
            pm = np.zeros(L_max, dtype=bool); pm[L:] = True
            lm = np.zeros(L_max, dtype=bool); lm[:L] = True
            bx.append(xp); by.append(yp); bp.append(pm); bl.append(lm)
            if with_impossible_pairs:
                for (ii, jj) in find_impossible_pairs(x):
                    pb.append(bi); pi_.append(ii); pj_.append(jj)
        batch = {
            'x': np.array(bx),
            'y': np.array(by),
            'padding_mask': np.array(bp),
            'loss_mask': np.array(bl),
        }
        if with_impossible_pairs:
            batch['pairs_b'] = np.array(pb, dtype=np.int64)
            batch['pairs_i'] = np.array(pi_, dtype=np.int64)
            batch['pairs_j'] = np.array(pj_, dtype=np.int64)
        yield batch


def list_cached_files(folder: str, limit: int | None = None) -> list[Path]:
    files = sorted(Path(folder).glob('*.npz'))
    if limit:
        files = files[:limit]
    return files


def train_val_split(files: list[Path], frac: float = 0.9, seed: int = 0):
    rng = np.random.default_rng(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * frac)
    return shuffled[:cut], shuffled[cut:]
