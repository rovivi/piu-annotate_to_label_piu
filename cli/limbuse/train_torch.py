#!/usr/bin/env python3
"""Train the PyTorch port of LimbSequenceTransformer.

Mirror of ``train_mlx.py`` against ``arch_torch.LimbSequenceTransformerTorch``.
Reads the same .npz cache produced by ``cache_chunks.py`` so a single dataset
serves all backends.

Device autopick: CUDA/ROCm > MPS > CPU. Override with ``--device``.

For AMD 6950 XT (gfx1030 / RDNA2):
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python cli/limbuse/train_torch.py ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Heavy imports are kept here; arch_torch lazy-imports torch internally so we
# do the same.
import torch
import torch.nn as nn

from piu_annotate.ml.arch_torch import build_model, pick_device, save_safetensors


MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 256
SINGLES_INPUT_DIM = 28
DOUBLES_INPUT_DIM = 33


_BRACKETABLE_SET = frozenset(
    (min(a, b), max(a, b)) for a, b in [
        [0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2],
        [4, 5], [3, 6],
        [5, 6], [5, 7], [6, 7], [8, 7], [8, 9], [9, 7],
    ]
)


def find_impossible_pairs(x_np: np.ndarray) -> list[tuple[int, int]]:
    arrow_pos = x_np[:, 0].astype(int)
    num_dp = x_np[:, 6].astype(int)
    n = len(x_np)
    out = []
    i = 0
    while i < n:
        k = int(num_dp[i])
        if k >= 2:
            grp = list(range(i, min(i + k, n)))
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    p1 = arrow_pos[grp[a]]
                    p2 = arrow_pos[grp[b]]
                    if (min(p1, p2), max(p1, p2)) not in _BRACKETABLE_SET:
                        out.append((grp[a], grp[b]))
            i += k
        else:
            i += 1
    return out


def make_chunks(x, y, max_len=MAX_SEQ_LEN, overlap=CHUNK_OVERLAP):
    if len(x) <= max_len:
        return [(x, y)]
    chunks = []
    stride = max_len - overlap
    for s in range(0, len(x) - max_len + 1, stride):
        chunks.append((x[s:s + max_len], y[s:s + max_len]))
    if (len(x) - max_len) % stride != 0:
        chunks.append((x[-max_len:], y[-max_len:]))
    return chunks


def _prev_limb_onehot(y: np.ndarray, first_label: int = 3) -> np.ndarray:
    n = len(y)
    prev = np.zeros((n, 4), dtype=np.float32)
    prev[0, min(first_label, 3)] = 1.0
    for i in range(1, n):
        lbl = int(y[i - 1])
        prev[i, min(lbl, 2)] = 1.0
    return prev


def make_chunks_with_prev(x, y, max_len=MAX_SEQ_LEN, overlap=CHUNK_OVERLAP):
    if len(x) <= max_len:
        prev = _prev_limb_onehot(y, first_label=3)
        return [(np.concatenate([x, prev], axis=1), y)]
    chunks = []
    stride = max_len - overlap
    for s in range(0, len(x) - max_len + 1, stride):
        cx = x[s:s + max_len]
        cy = y[s:s + max_len]
        first = 3 if s == 0 else min(int(y[s - 1]), 2)
        prev = _prev_limb_onehot(cy, first_label=first)
        chunks.append((np.concatenate([cx, prev], axis=1), cy))
    if (len(x) - max_len) % stride != 0:
        s = len(x) - max_len
        cx = x[s:]
        cy = y[s:]
        first = 3 if s == 0 else min(int(y[s - 1]), 2)
        prev = _prev_limb_onehot(cy, first_label=first)
        chunks.append((np.concatenate([cx, prev], axis=1), cy))
    return chunks


def load_train_data_pair(files):
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


def compute_loss(logits, y, loss_mask, smoothing=0.1, class_weights=None):
    """Cross-entropy with label smoothing + optional class weights.

    Padding tokens (y == -1) are masked out. ``y`` is float for compat with
    the numpy batch arrays; cast to long internally."""
    B, L, C = logits.shape
    logits_flat = logits.reshape(B * L, C)
    y_long = y.reshape(B * L).clamp_min(0).long()
    valid = (loss_mask & (y >= 0)).reshape(B * L).to(logits.dtype)

    log_softmax = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    nll = -log_softmax.gather(1, y_long.unsqueeze(-1)).squeeze(-1)
    if smoothing > 0:
        uniform = -log_softmax.mean(dim=-1)
        nll = (1.0 - smoothing) * nll + smoothing * uniform
    if class_weights is not None:
        w = class_weights[y_long.clamp(0, C - 1)]
        nll = nll * w
    return (nll * valid).sum() / valid.sum().clamp_min(1e-8)


def compute_impossible_penalty(logits, pb, pi, pj):
    """Penalize same-foot probability mass on impossible bracket pairs.

    Loss term = mean over pairs of (P(L_i)P(L_j) + P(R_i)P(R_j))."""
    B, L, C = logits.shape
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    flat = log_softmax.reshape(B * L, C)
    flat_i = pb * L + pi
    flat_j = pb * L + pj
    lp_i = flat[flat_i]
    lp_j = flat[flat_j]
    p_i = lp_i.exp()
    p_j = lp_j.exp()
    pen = (p_i[:, 0] * p_j[:, 0] + p_i[:, 1] * p_j[:, 1]).sum()
    return pen / (flat_i.shape[0] + 1e-8)


def compute_accuracy(logits, y, loss_mask):
    preds = logits.argmax(dim=-1)
    target = y.long()
    mask = (loss_mask & (y >= 0))
    correct = ((preds == target) & mask).sum()
    total = mask.sum()
    return int(correct.item()), int(total.item())


def evaluate(model, val_chunks, batch_size, device):
    model.eval()
    loss_sum, n_batches = 0.0, 0
    total_correct, total_tokens = 0, 0
    class_correct = np.zeros(3, dtype=np.int64)
    class_total = np.zeros(3, dtype=np.int64)
    with torch.inference_mode():
        for batch in build_batches(val_chunks, batch_size, shuffle=False):
            x = torch.from_numpy(batch['x']).to(device)
            y = torch.from_numpy(batch['y']).to(device)
            pm = torch.from_numpy(batch['padding_mask']).to(device)
            lm = torch.from_numpy(batch['loss_mask']).to(device)
            logits = model(x, pm)
            loss = compute_loss(logits, y, lm)
            c, t = compute_accuracy(logits, y, lm)
            loss_sum += float(loss.item())
            total_correct += c
            total_tokens += t
            n_batches += 1
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_np = y.long().cpu().numpy()
            lm_np = lm.cpu().numpy()
            for c_idx in range(3):
                m = lm_np & (y_np == c_idx)
                class_total[c_idx] += m.sum()
                class_correct[c_idx] += (preds[m] == c_idx).sum()
    pc = {c: class_correct[c] / max(class_total[c], 1) for c in range(3)}
    return {
        'loss': loss_sum / max(n_batches, 1),
        'acc': total_correct / max(total_tokens, 1),
        'per_class': pc,
        'class_total': class_total,
    }


def evaluate_ar_accuracy(model, val_chunks, batch_size, device):
    """2-pass AR accuracy. Pass 1 with teacher forcing, pass 2 substitutes
    pass-1 predictions into the prev_limb feature slot."""
    model.eval()
    correct_total, n_total = 0, 0
    with torch.inference_mode():
        for batch in build_batches(val_chunks, batch_size, shuffle=False):
            x = torch.from_numpy(batch['x']).to(device)
            y = torch.from_numpy(batch['y']).to(device)
            pm = torch.from_numpy(batch['padding_mask']).to(device)
            lm = torch.from_numpy(batch['loss_mask']).to(device)
            logits1 = model(x, pm)
            preds1 = logits1.argmax(dim=-1)
            B, L, _ = x.shape
            if L > 1:
                x_ar = x.clone()
                onehot = torch.zeros(B, L - 1, 4, device=device, dtype=x.dtype)
                onehot.scatter_(2, preds1[:, :-1].clamp(0, 2).unsqueeze(-1), 1.0)
                x_ar[:, 1:, -4:] = onehot
            else:
                x_ar = x
            logits2 = model(x_ar, pm)
            c, t = compute_accuracy(logits2, y, lm)
            correct_total += c
            n_total += t
    return correct_total / max(n_total, 1)


def lr_lambda_factory(total_steps: int, warmup_steps: int):
    """Linear warmup then cosine decay to ~end-of-schedule."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def train(
    folder: str,
    sd: str,
    out_dir: str,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 3e-4,
    seed: int = 0,
    warmup_epochs: int = 2,
    patience: int = 8,
    limit_charts: int | None = None,
    label_smoothing: float = 0.1,
    class_weight_e: float = 1.0,
    large_model: bool = False,
    resume_from: str | None = None,
    impossible_penalty: float = 0.0,
    device: str | None = None,
    use_compile: bool = False,
    grad_accum_steps: int = 1,
    dtype: str = 'fp32',
    save_by_ar: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    dev = pick_device(prefer=device)
    logger.info(f'Device: {dev}')

    all_files = sorted(Path(folder).glob('*.npz'))
    if limit_charts:
        all_files = all_files[:limit_charts]
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.9)
    train_files, val_files = shuffled[:split], shuffled[split:]

    train_data = load_train_data_pair(train_files)
    val_chunks = load_val_chunks(val_files)
    logger.info(f'Train files: {len(train_files)}  Val chunks: {len(val_chunks)}')

    input_dim_raw = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM

    if limit_charts and limit_charts <= 200:
        logger.info('SMOKE: small model')
        d_model, n_heads, n_layers, ffn_dim = 64, 4, 2, 128
    elif large_model:
        logger.info('LARGE model: d_model=384, n_layers=8')
        d_model, n_heads, n_layers, ffn_dim = 384, 8, 8, 1536
    else:
        d_model, n_heads, n_layers, ffn_dim = 256, 8, 6, 1024

    model = build_model(
        input_dim=input_dim_raw,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
    ).to(dev)

    if resume_from and os.path.exists(resume_from):
        from piu_annotate.ml.arch_torch import load_safetensors
        load_safetensors(model, resume_from, device=dev)
        logger.info(f'Resumed from {resume_from}')

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model: {n_params:,} parameters')

    class_weights_t = None
    if class_weight_e != 1.0:
        class_weights_t = torch.tensor([1.0, 1.0, class_weight_e], device=dev)

    dummy_chunks = build_both_epoch_chunks(train_data)
    steps_per_epoch = max(len(dummy_chunks) // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    del dummy_chunks
    logger.info(f'~{steps_per_epoch} steps/epoch, {total_steps} total, {warmup_steps} warmup')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(total_steps, warmup_steps),
    )

    use_imp = impossible_penalty > 0.0
    fwd = model
    if use_compile:
        try:
            fwd = torch.compile(model)
            logger.info('torch.compile enabled')
        except Exception as e:
            logger.warning(f'torch.compile failed, falling back to eager: {e}')

    # Mixed precision setup. For low-VRAM AMD/CUDA: bf16 cuts ~40% memory at
    # no acc cost; fp16 cuts ~50% but needs a GradScaler. bf16 is preferred
    # on RDNA2/3 + recent CUDA.
    autocast_dtype = {
        'fp32': None,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
    }.get(dtype, None)
    scaler = None
    if dtype == 'fp16' and dev.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    if autocast_dtype is not None:
        logger.info(f'Mixed precision: {dtype}')
    if grad_accum_steps > 1:
        logger.info(f'Gradient accumulation: {grad_accum_steps} steps (effective batch = {batch_size * grad_accum_steps})')

    best_acc = 0.0
    no_improve = 0
    gstep = 0

    for epoch in range(epochs):
        logger.info(f'=== Epoch {epoch+1}/{epochs} ===')
        train_chunks = build_both_epoch_chunks(train_data)
        model.train()
        epoch_loss = 0.0
        nb = 0
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        for batch in tqdm(
            build_batches(train_chunks, batch_size, shuffle=True, seed=seed + epoch,
                          with_impossible_pairs=use_imp),
            desc='Training', leave=False,
        ):
            x = torch.from_numpy(batch['x']).to(dev)
            y = torch.from_numpy(batch['y']).to(dev)
            pm = torch.from_numpy(batch['padding_mask']).to(dev)
            lm = torch.from_numpy(batch['loss_mask']).to(dev)

            if autocast_dtype is not None:
                ctx = torch.autocast(device_type=dev.type, dtype=autocast_dtype)
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                logits = fwd(x, pm)
                loss = compute_loss(logits, y, lm, smoothing=label_smoothing,
                                    class_weights=class_weights_t)
            if use_imp and len(batch['pairs_b']) > 0:
                pb = torch.from_numpy(batch['pairs_b']).to(dev)
                pi = torch.from_numpy(batch['pairs_i']).to(dev)
                pj = torch.from_numpy(batch['pairs_j']).to(dev)
                loss = loss + impossible_penalty * compute_impossible_penalty(
                    logits, pb, pi, pj
                )
            loss_to_log = float(loss.item())
            # Scale loss for accumulation so the optimizer step matches the
            # full effective batch size's gradient magnitude.
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            if accum_count >= grad_accum_steps:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

            epoch_loss += loss_to_log
            if gstep % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info(f'  step {gstep}, loss={loss_to_log:.4f}, lr={lr_now:.2e}')
            nb += 1
            gstep += 1

        # Flush any leftover grads at end of epoch
        if accum_count > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        train_loss = epoch_loss / max(nb, 1)
        metrics = evaluate(model, val_chunks, batch_size, dev)
        pc = metrics['per_class']
        ct = metrics['class_total']
        logger.info(
            f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={metrics['loss']:.4f}, val_acc={metrics['acc']*100:.1f}%, "
            f"L={pc[0]*100:.1f}%(n={ct[0]}) R={pc[1]*100:.1f}%(n={ct[1]}) "
            f"E={pc[2]*100:.1f}%(n={ct[2]})"
        )
        ar_acc = evaluate_ar_accuracy(model, val_chunks, batch_size, dev)
        logger.info(f'  AR val_acc={ar_acc*100:.1f}%  (oracle gap={(metrics["acc"]-ar_acc)*100:.1f}pp)')

        # Save best by AR accuracy (realistic metric) or teacher-forced val_acc.
        track_acc = ar_acc if save_by_ar else metrics['acc']
        if track_acc > best_acc:
            best_acc = track_acc
            no_improve = 0
            best_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-torch-best.safetensors')
            save_safetensors(model, best_path)
            with open(os.path.join(out_dir, f'{sd}-arrows_to_limb-torch-best.meta'), 'w') as f:
                json.dump({
                    'input_dim': input_dim_raw,
                    'd_model': d_model, 'n_heads': n_heads,
                    'n_layers': n_layers, 'ffn_dim': ffn_dim,
                    'n_classes': 3,
                }, f)
            metric_label = 'AR val_acc' if save_by_ar else 'val_acc'
            logger.success(f'New best saved: {metric_label}={best_acc*100:.1f}%')
        else:
            no_improve += 1
            logger.info(f'  No improvement {no_improve}/{patience}')
            if no_improve >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    final_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-torch-final.safetensors')
    save_safetensors(model, final_path)
    logger.success(f'Training complete. Best val_acc: {best_acc*100:.1f}%')

    with open(os.path.join(out_dir, 'train_config.json'), 'w') as f:
        json.dump({
            'backend': 'torch', 'sd': sd, 'epochs': epochs, 'batch_size': batch_size,
            'lr': lr, 'seed': seed, 'input_dim': input_dim_raw + 4,
            'best_acc': best_acc, 'd_model': d_model, 'n_heads': n_heads,
            'n_layers': n_layers, 'ffn_dim': ffn_dim,
        }, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--singles_or_doubles', required=True, choices=['singles', 'doubles'])
    p.add_argument('--manual_chart_struct_folder', required=True,
                   help='Directory of .npz files produced by cache_chunks.py')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup_epochs', type=int, default=2)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--limit_charts', type=int, default=None)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--class_weight_e', type=float, default=1.0)
    p.add_argument('--large_model', action='store_true')
    p.add_argument('--resume_from', default=None)
    p.add_argument('--impossible_penalty', type=float, default=0.0)
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None)
    p.add_argument('--compile', dest='use_compile', action='store_true')
    p.add_argument('--grad_accum_steps', type=int, default=1,
                   help='Gradient accumulation. Effective batch = batch_size * grad_accum_steps. '
                        'Use to fit larger effective batches on small VRAM.')
    p.add_argument('--dtype', choices=['fp32', 'bf16', 'fp16'], default='fp32',
                   help='Mixed precision. bf16 saves ~40%% VRAM, fp16 saves ~50%% (needs CUDA + GradScaler).')
    p.add_argument('--save_by_ar', action='store_true',
                   help='Save best model by autoregressive accuracy instead of teacher-forced val_acc. '
                        'More realistic: picks the checkpoint that performs best in real inference.')
    args = p.parse_args()

    train(
        folder=args.manual_chart_struct_folder, sd=args.singles_or_doubles,
        out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, warmup_epochs=args.warmup_epochs,
        patience=args.patience, limit_charts=args.limit_charts,
        label_smoothing=args.label_smoothing, class_weight_e=args.class_weight_e,
        large_model=args.large_model, resume_from=args.resume_from,
        impossible_penalty=args.impossible_penalty, device=args.device,
        use_compile=args.use_compile,
        grad_accum_steps=args.grad_accum_steps, dtype=args.dtype,
        save_by_ar=args.save_by_ar,
    )


if __name__ == '__main__':
    main()
