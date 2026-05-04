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
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.featurizers import ChartStructFeaturizer
from piu_annotate.formats.mirror import mirror_chartstruct
from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer

MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 128

SINGLES_INPUT_DIM = 18
DOUBLES_INPUT_DIM = 23

def save_model(path: str, model: nn.Module) -> None:
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(path, flat)

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

def load_chunks_from_files(files: list[Path], mirror_prob: float, rng: np.random.Generator):
    all_chunks = []
    n_mirror = 0
    for f in tqdm(files, desc=f'Loading cached'):
        d = np.load(f)
        if mirror_prob > 0 and rng.random() < mirror_prob:
            x, y = d['x_mirror'], d['y_mirror']
            n_mirror += 1
        else:
            x, y = d['x'], d['y']
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        for cx, cy in make_chunks(x, y):
            all_chunks.append((cx, cy))
    logger.info(f'Loaded {len(all_chunks)} chunks ({n_mirror}/{len(files)} mirrored)')
    return all_chunks

def build_batches(chunks, batch_size, shuffle=True, seed=0):
    sorted_idx = sorted(range(len(chunks)), key=lambda i: chunks[i][0].shape[0])
    batches = [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    for batch_idxs in batches:
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
    
    from functools import partial
    
    def loss_fn(mdl, x, y, pm, lm):
        return compute_loss(mdl(x, pm), y, lm)
        
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, pm, lm):
        loss, grads = loss_and_grad_fn(model, x, y, pm, lm)
        optimizer.update(model, grads)
        return loss
    
    for batch in tqdm(build_batches(train_chunks, batch_size, shuffle=True), desc='Training', leave=False):
        x = mx.array(batch['x'])
        y = mx.array(batch['y'])
        pm = mx.array(batch['padding_mask'])
        lm = mx.array(batch['loss_mask'])
        
        loss = step(x, y, pm, lm)
        if n_batches % 8 == 0:
            mx.eval(model.parameters(), optimizer.state)
        epoch_loss += float(loss)
        if n_batches % 50 == 0:
            logger.info(f'  step {n_batches}, loss={float(loss):.4f}')
        n_batches += 1
    mx.eval(model.parameters(), optimizer.state)
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
    rng = np.random.default_rng(seed)

    os.makedirs(out_dir, exist_ok=True)

    all_files = sorted(Path(folder).glob('*.npz'))
    if limit_charts:
        all_files = all_files[:limit_charts]
        
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.9)
    train_files, val_files = shuffled[:split], shuffled[split:]

    train_chunks = load_chunks_from_files(train_files, mirror_prob=0.5, rng=rng)
    val_chunks   = load_chunks_from_files(val_files,   mirror_prob=0.0, rng=rng)
    logger.info(f"Train chunks: {len(train_chunks)}, Val chunks: {len(val_chunks)}")

    input_dim = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM
    
    # if limit charts < 200, assume it is a smoke test, use smaller model.
    if limit_charts and limit_charts <= 200:
        logger.info("SMOKE TEST: Using smaller dummy model.")
        model = LimbSequenceTransformer(input_dim=input_dim, d_model=64, n_heads=4, n_layers=2, ffn_dim=128)
        d_model_save, n_heads_save, n_layers_save, ffn_dim_save = 64, 4, 2, 128
    else:
        model = LimbSequenceTransformer(input_dim=input_dim, d_model=128, n_heads=8, n_layers=4, ffn_dim=512)
        d_model_save, n_heads_save, n_layers_save, ffn_dim_save = 128, 8, 4, 512
        
    mx.eval(model.parameters())
    logger.info("Model built")

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    best_acc = 0.0
    total_steps = len(train_chunks) // batch_size * epochs
    warmup_steps = len(train_chunks) // batch_size * warmup_epochs

    for epoch in range(epochs):
        logger.info(f'=== Epoch {epoch+1}/{epochs} starting ===')
        lr_now = cosine_schedule(epoch * len(train_chunks) // batch_size, total_steps, lr, warmup_steps)
        optimizer.learning_rate = lr_now

        train_loss = train_epoch(model, train_chunks, optimizer, batch_size)
        val_metrics = evaluate(model, val_chunks, batch_size)

        logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']*100:.1f}%, lr={lr_now:.2e}")

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            save_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
            save_model(save_path, model)
            with open(os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.meta'), 'w') as f:
                json.dump({'input_dim': input_dim, 'd_model': d_model_save, 'n_heads': n_heads_save, 'n_layers': n_layers_save, 'ffn_dim': ffn_dim_save, 'n_classes': 3}, f)
            logger.success(f"New best model saved: acc={best_acc*100:.1f}%")

    final_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-final.safetensors')
    save_model(final_path, model)
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