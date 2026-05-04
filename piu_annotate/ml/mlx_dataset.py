from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import Iterator
from loguru import logger

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.featurizers import ChartStructFeaturizer
from piu_annotate.ml.datapoints import LimbLabel
from piu_annotate.formats.mirror import mirror_chartstruct


MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 128


def chartstruct_to_sequence(cs: ChartStruct) -> tuple[np.ndarray, np.ndarray]:
    ft = ChartStructFeaturizer(cs)
    x_raw = ft.pt_array
    cmf = np.tile(ft.chart_metadata_features, (len(x_raw), 1))
    x = np.concatenate([x_raw, cmf], axis=1)
    labels_raw = ft.get_labels_from_limb_col('Limb annotation')
    return x, labels_raw


def make_chunks(
    x: np.ndarray,
    y: np.ndarray,
    max_len: int = MAX_SEQ_LEN,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(x) <= max_len:
        return [(x, y)]
    chunks = []
    stride = max_len - overlap
    for start in range(0, len(x) - max_len + 1, stride):
        chunks.append((x[start:start + max_len], y[start:start + max_len]))
    if len(x) % stride != 0:
        chunks.append((x[-max_len:], y[-max_len:]))
    return chunks


class MLXDataset:
    def __init__(
        self,
        folder: str,
        sd: str,
        mirror_prob: float = 0.5,
        max_len: int = MAX_SEQ_LEN,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.folder = folder
        self.sd = sd
        self.mirror_prob = mirror_prob
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        self.files = [f for f in Path(folder).glob('*.csv')
                      if ChartStruct.singles_or_doubles(ChartStruct.from_file(str(f)).df) == sd
                      or (lambda: (lambda cs: cs.singles_or_doubles() == sd)(ChartStruct.from_file(str(f))))()]

        self._all_chunks: list[tuple[np.ndarray, np.ndarray, bool]] = []
        self._load_chunks()
        if shuffle:
            self.rng.shuffle(self._all_chunks)

    def _load_chunks(self) -> None:
        for f in logger.progress(self.files, desc='Loading charts'):
            try:
                cs = ChartStruct.from_file(str(f))
                x, y = chartstruct_to_sequence(cs)
                chunks = make_chunks(x, y, self.max_len, CHUNK_OVERLAP)
                for cx, cy in chunks:
                    self._all_chunks.append((cx, cy, False))
            except Exception as e:
                logger.warning(f'Error loading {f}: {e}')

    def __len__(self) -> int:
        return len(self._all_chunks)

    def __getitem__(self, idx: int) -> dict:
        x, y, was_mirrored = self._all_chunks[idx]
        if self.mirror_prob > 0 and self.rng.random() < self.mirror_prob:
            x = x.copy()
            mirrored_line = mirror_chartstruct(ChartStruct.from_file(str(self.files[idx % len(self.files)])))
            x_m, y_m = chartstruct_to_sequence(mirrored_line)
            x[:len(x_m)] = x_m
            y[:len(y_m)] = y_m
            was_mirrored = True

        padding_mask = np.zeros(len(x), dtype=bool)
        loss_mask = np.ones(len(x), dtype=bool)
        return {
            'x': x.astype(np.float32),
            'padding_mask': padding_mask,
            'y': y.astype(np.float32),
            'loss_mask': loss_mask,
            'was_mirrored': was_mirrored,
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        max_len = max(b['x'].shape[0] for b in batch)
        B = len(batch)
        x = np.zeros((B, max_len, batch[0]['x'].shape[1]), dtype=np.float32)
        padding_mask = np.ones((B, max_len), dtype=bool)
        y = np.full((B, max_len), -1, dtype=np.float32)
        loss_mask = np.zeros((B, max_len), dtype=bool)
        was_mirrored = np.zeros(B, dtype=bool)
        for i, b in enumerate(batch):
            L = b['x'].shape[0]
            x[i, :L] = b['x']
            padding_mask[i, :L] = b['padding_mask']
            y[i, :L] = b['y']
            loss_mask[i, :L] = b['loss_mask']
            was_mirrored[i] = b['was_mirrored']
        return {'x': x, 'padding_mask': padding_mask, 'y': y, 'loss_mask': loss_mask, 'was_mirrored': was_mirrored}


def train_val_split(dataset: MLXDataset, val_frac: float = 0.1, seed: int = 0):
    n = len(dataset)
    n_val = int(n * val_frac)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return train_indices, val_indices


def build_loaders(folder: str, sd: str, batch_size: int = 16,
                  mirror_prob: float = 0.5, val_frac: float = 0.1,
                  seed: int = 0):
    dataset = MLXDataset(folder, sd, mirror_prob=mirror_prob, shuffle=True, seed=seed)
    train_idxs, val_idxs = train_val_split(dataset, val_frac, seed)
    train_chunks = [dataset._all_chunks[i] for i in train_idxs]
    val_chunks = [dataset._all_chunks[i] for i in val_idxs]
    return train_chunks, val_chunks, dataset