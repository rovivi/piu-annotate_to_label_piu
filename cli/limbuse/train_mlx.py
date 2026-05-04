#!/usr/bin/env python3
"""
Train the LimbSequenceTransformer on ChartStruct CSV data.
Processes charts as full sequences with bidirectional attention.
"""
from __future__ import annotations
import os
import sys
import json
import math
import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.featurizers import ChartStructFeaturizer
from piu_annotate.formats.mirror import mirror_chartstruct


MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 128


SINGLES_INPUT_DIM = 18
DOUBLES_INPUT_DIM = 21


def _flatten_params(d: dict, parent_key: str = '') -> dict[str, np.ndarray]:
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}.{k}' if parent_key else k
        if isinstance(v, list):
            for i, elem in enumerate(v):
                items.update(_flatten_params(elem, f'{new_key}.{i}'))
        elif isinstance(v, dict):
            items.update(_flatten_params(v, new_key))
        else:
            items[new_key] = np.array(v)
    return items


def sinusoidal_pos_encoding(seq_len: int, d_model: int) -> mx.array:
    positions = mx.arange(seq_len).reshape(-1, 1)
    div_term = mx.exp(mx.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = mx.concatenate([mx.sin(positions * div_term), mx.cos(positions * div_term)], axis=1)
    if d_model % 2 == 1:
        pe = mx.concatenate([pe, mx.zeros((seq_len, 1))], axis=1)
    return pe


class LimbSequenceTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 512,
        max_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = sinusoidal_pos_encoding(max_len, d_model)
        self.encoder = nn.TransformerEncoder(
            num_layers=n_layers,
            dims=d_model,
            num_heads=n_heads,
            mlp_dims=ffn_dim,
            dropout=dropout,
            checkpoint=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_head = nn.Linear(d_model, 3)
        self.d_model = d_model

    def __call__(self, x: mx.array, padding_mask: mx.array) -> mx.array:
        B, L, _ = x.shape
        h = self.input_proj(x)
        h = h + self.pos_enc[:L][None, :, :]
        h = self.dropout(h)
        if mx.sum(padding_mask) > 0:
            attn_mask = mx.where(padding_mask[:, None, :, None], -1e9, 0.0)
        else:
            attn_mask = None
        h = self.encoder(h, mask=attn_mask)
        return self.out_head(h)


def _build_padding_mask(padding_mask: mx.array) -> mx.array:
    return mx.where(padding_mask[:, None, :], -1e9, 0.0)


def chartstruct_to_sequence(cs: ChartStruct) -> tuple[np.ndarray, np.ndarray]:
    ft = ChartStructFeaturizer(cs)
    x_raw = np.stack(ft.pt_array)
    cmf = np.tile(ft.chart_metadata_features, (len(x_raw), 1))
    x = np.concatenate([x_raw, cmf], axis=1)
    labels_raw = ft.get_labels_from_limb_col('Limb annotation')
    return x.astype(np.float32), labels_raw.astype(np.float32)


def make_chunks(x, y, max_len=MAX_SEQ_LEN, overlap=CHUNK_OVERLAP):
    if len(x) <= max_len:
        return [(x, y)]
    chunks = []
    stride = max_len - overlap
    for start in range(0, len(x) - max_len + 1, stride):
        chunks.append((x[start:start + max_len], y[start:start + max_len]))
    if (len(x) - max_len) % stride != 0:
        chunks.append((x[-max_len:], y[-max_len:]))
    return chunks


def load_all_chunks(folder: str, sd: str, mirror_prob: float, seed: int, limit: int | None = None):
    files = list(Path(folder).glob('*.csv'))
    if limit:
        files = files[:limit]
    rng = np.random.default_rng(seed)
    all_chunks = []
    counts = {'total': 0, 'mirrored': 0}
    for f in tqdm(files, desc=f'Loading {sd}'):
        try:
            cs = ChartStruct.from_file(str(f))
            if cs.singles_or_doubles() != sd:
                continue
            x, y = chartstruct_to_sequence(cs)
            chunks = make_chunks(x, y)
            for cx, cy in chunks:
                was_mirrored = False
                if mirror_prob > 0 and rng.random() < mirror_prob:
                    mirrored_cs = mirror_chartstruct(cs)
                    mx2, my = chartstruct_to_sequence(mirrored_cs)
                    min_len = min(len(cx), len(mx2))
                    cx[:min_len] = mx2[:min_len]
                    cy[:min_len] = my[:min_len]
                    was_mirrored = True
                counts['total'] += 1
                if was_mirrored:
                    counts['mirrored'] += 1
                all_chunks.append((cx, cy))
        except Exception as e:
            logger.warning(f'Error loading {f}: {e}')
    logger.info(f"Loaded {counts['total']} chunks, {counts['mirrored']} mirrored ({counts['mirrored']/max(counts['total'],1)*100:.1f}%)")
    return all_chunks


def build_batches(chunks, batch_size, shuffle=True, seed=0):
    indices = np.arange(len(chunks))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idxs = indices[start:start + batch_size]
        batch_x = []
        batch_y = []
        batch_pad = []
        batch_loss = []
        max_len = max(chunks[i][0].shape[0] for i in batch_idxs)
        for i in batch_idxs:
            x, y = chunks[i]
            L = x.shape[0]
            pad = np.zeros(L, dtype=bool)
            loss_mask = np.ones(L, dtype=bool)
            x_padded = np.zeros((max_len, x.shape[1]), dtype=np.float32)
            y_padded = np.full((max_len,), -1, dtype=np.float32)
            x_padded[:L] = x
            y_padded[:L] = y
            batch_x.append(x_padded)
            batch_y.append(y_padded)
            batch_pad.append(np.zeros(max_len, dtype=bool))
            batch_loss.append(np.zeros(max_len, dtype=bool))
            batch_loss[-1][:L] = True
        yield {
            'x': np.array(batch_x),
            'y': np.array(batch_y),
            'padding_mask': np.array(batch_pad),
            'loss_mask': np.array(batch_loss),
        }


def compute_loss(logits, y, loss_mask):
    B, L, C = logits.shape
    logits_flat = logits.reshape(B * L, C)
    y_flat = y.reshape(B * L).astype(mx.int32)
    valid_mask = (loss_mask & (y >= 0)).astype(logits.dtype).reshape(B * L)
    valid_count = mx.sum(valid_mask) + 1e-8
    max_logits = mx.max(logits_flat, axis=-1, keepdims=True)
    shifted = logits_flat - max_logits
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    log_softmax = shifted - log_sum_exp
    nll = -mx.take_along_axis(log_softmax, y_flat[:, None], axis=1).squeeze(-1)
    weighted_loss = nll * valid_mask
    return mx.sum(weighted_loss) / valid_count


def compute_accuracy(logits, y, loss_mask):
    B, L, C = logits.shape
    preds = mx.argmax(logits, axis=-1)
    target = y.astype(mx.int32)
    mask = (loss_mask & (y >= 0)).astype(mx.int32)
    correct = ((preds == target) & (mask == 1))
    return float(mx.sum(correct) / (mx.sum(mask) + 1e-8))


def evaluate(model, val_chunks, batch_size):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for batch in build_batches(val_chunks, batch_size, shuffle=False):
        x = mx.array(batch['x'])
        y = mx.array(batch['y'])
        pm = mx.array(batch['padding_mask'])
        lm = mx.array(batch['loss_mask'])
        logits = model(x, pm)
        loss = compute_loss(logits, y, lm)
        acc = compute_accuracy(logits, y, lm)
        total_loss += float(loss)
        total_acc += acc
        n_batches += 1
    return {'loss': total_loss / max(n_batches, 1), 'acc': total_acc / max(n_batches, 1)}


def train_epoch(model, train_chunks, optimizer, batch_size, grad_clip=1.0):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y, pm, lm: compute_loss(m(x, pm), y, lm))
    for batch in tqdm(build_batches(train_chunks, batch_size, shuffle=True), desc='Training', leave=False):
        x = mx.array(batch['x'])
        y = mx.array(batch['y'])
        pm = mx.array(batch['padding_mask'])
        lm = mx.array(batch['loss_mask'])
        loss, grads = loss_and_grad(model, x, y, pm, lm)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        epoch_loss += float(loss)
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


def cosine_schedule(step, total, lr_max, warmup_steps):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / (total - warmup_steps)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    folder: str,
    sd: str,
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 3e-4,
    seed: int = 0,
    warmup_epochs: int = 1,
    limit_charts: int | None = None,
):
    mx.random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    train_chunks = load_all_chunks(folder, sd, mirror_prob=0.5, seed=seed, limit=limit_charts)
    val_chunks = load_all_chunks(folder, sd, mirror_prob=0.0, seed=seed+1, limit=limit_charts)
    logger.info(f"Train chunks: {len(train_chunks)}, Val chunks: {len(val_chunks)}")

    input_dim = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM
    model = LimbSequenceTransformer(input_dim=input_dim, d_model=128, n_heads=8, n_layers=4, ffn_dim=512)
    mx.eval(model.parameters())
    logger.info("Model built")

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    best_acc = 0.0
    total_steps = len(train_chunks) // batch_size * epochs
    warmup_steps = len(train_chunks) // batch_size * warmup_epochs

    for epoch in range(epochs):
        lr_now = cosine_schedule(epoch * len(train_chunks) // batch_size, total_steps, lr, warmup_steps)
        optimizer.learning_rate = lr_now

        train_loss = train_epoch(model, train_chunks, optimizer, batch_size)
        val_metrics = evaluate(model, val_chunks, batch_size)

        logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']*100:.1f}%, lr={lr_now:.2e}")

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            save_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
            state = model.parameters()
            flat_state = _flatten_params(state)
            np.savez(save_path, **flat_state)
            with open(os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.meta'), 'w') as f:
                json.dump({'input_dim': input_dim, 'd_model': 128, 'n_heads': 8, 'n_layers': 4, 'ffn_dim': 512, 'n_classes': 3}, f)
            logger.success(f"New best model saved: acc={best_acc*100:.1f}%")

    final_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-final.safetensors')
    flat_state = _flatten_params(model.parameters())
    np.savez(final_path, **flat_state)
    logger.success(f"Training complete. Best acc: {best_acc*100:.1f}%")

    config = {
        'sd': sd, 'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
        'seed': seed, 'input_dim': input_dim, 'best_acc': best_acc,
    }
    with open(os.path.join(out_dir, 'train_config.json'), 'w') as f:
        json.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--singles_or_doubles', type=str, required=True)
    parser.add_argument('--manual_chart_struct_folder', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit_charts', type=int, default=None)
    args = parser.parse_args()

    train(
        folder=args.manual_chart_struct_folder,
        sd=args.singles_or_doubles,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        limit_charts=args.limit_charts,
    )