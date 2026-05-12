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
from functools import partial
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.optimizers import linear_schedule, cosine_decay, join_schedules, clip_grad_norm as mx_clip_grad_norm
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer

MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 256  # was 128; larger overlap improves boundary predictions

_BRACKETABLE_SET: frozenset[tuple[int, int]] = frozenset(
    (min(a, b), max(a, b)) for a, b in [
        [0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2],  # P1 center + side brackets
        [4, 5], [3, 6],                                    # inter-pad
        [5, 6], [5, 7], [6, 7], [8, 7], [8, 9], [9, 7],  # P2 center + side brackets
    ]
)


def find_impossible_pairs(x_np: np.ndarray) -> list[tuple[int, int]]:
    """Return (i, j) index pairs in x that would be impossible same-foot brackets."""
    arrow_pos = x_np[:, 0].astype(int)
    num_dp = x_np[:, 6].astype(int)
    N = len(x_np)
    pairs: list[tuple[int, int]] = []
    i = 0
    while i < N:
        n = int(num_dp[i])
        if n >= 2:
            group = list(range(i, min(i + n, N)))
            for g1 in range(len(group)):
                for g2 in range(g1 + 1, len(group)):
                    p1 = arrow_pos[group[g1]]
                    p2 = arrow_pos[group[g2]]
                    if (min(p1, p2), max(p1, p2)) not in _BRACKETABLE_SET:
                        pairs.append((group[g1], group[g2]))
            i += n
        else:
            i += 1
    return pairs


def compute_impossible_penalty(
    logits: 'mx.array',
    pairs_b: 'mx.array',
    pairs_i: 'mx.array',
    pairs_j: 'mx.array',
) -> 'mx.array':
    """Penalize impossible same-foot bracket predictions.

    For each impossible pair (b, i, j): penalise P(L_i)*P(L_j) + P(R_i)*P(R_j).
    """
    B, L, C = logits.shape
    max_l = mx.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_l - mx.log(mx.sum(mx.exp(logits - max_l), axis=-1, keepdims=True))
    log_probs_flat = log_probs.reshape(B * L, C)

    flat_i = pairs_b * L + pairs_i
    flat_j = pairs_b * L + pairs_j

    lp_i = mx.take(log_probs_flat, flat_i, axis=0)  # (P, 3)
    lp_j = mx.take(log_probs_flat, flat_j, axis=0)  # (P, 3)

    probs_i = mx.exp(lp_i)
    probs_j = mx.exp(lp_j)

    penalty = mx.sum(probs_i[:, 0] * probs_j[:, 0] + probs_i[:, 1] * probs_j[:, 1])
    return penalty / (flat_i.shape[0] + 1e-8)

SINGLES_INPUT_DIM = 28   # 24 arrow features + 4 prev-limb one-hot (L, R, E, START)
DOUBLES_INPUT_DIM = 33   # 29 arrow features + 4 prev-limb one-hot


def save_model(path: str, model: nn.Module) -> None:
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(path, flat)


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


def _prev_limb_onehot(y: np.ndarray, first_label: int = 3) -> np.ndarray:
    """Build one-hot previous-limb feature: shape (N, 4) for classes [L, R, E, START].

    prev[0] = one-hot(first_label)  — START(3) at chart beginning, or actual label
              for the first token of overlapping chunks.
    prev[i] = one-hot(y[i-1])       — teacher-forced ground truth for training.
    """
    N = len(y)
    prev = np.zeros((N, 4), dtype=np.float32)
    prev[0, min(first_label, 3)] = 1.0
    for i in range(1, N):
        lbl = int(y[i - 1])
        prev[i, min(lbl, 2)] = 1.0
    return prev


def make_chunks_with_prev(x, y, max_len=MAX_SEQ_LEN, overlap=CHUNK_OVERLAP):
    """Like make_chunks but appends a 4-dim prev-limb one-hot to every token.

    For the first token of each chunk the previous label is taken from the
    full y array so that overlapping chunks have correct cross-boundary context.
    """
    if len(x) <= max_len:
        prev = _prev_limb_onehot(y, first_label=3)
        return [(np.concatenate([x, prev], axis=1), y)]

    chunks = []
    stride = max_len - overlap

    for start in range(0, len(x) - max_len + 1, stride):
        cx = x[start:start + max_len]
        cy = y[start:start + max_len]
        first = 3 if start == 0 else min(int(y[start - 1]), 2)
        prev = _prev_limb_onehot(cy, first_label=first)
        chunks.append((np.concatenate([cx, prev], axis=1), cy))

    if (len(x) - max_len) % stride != 0:
        start = len(x) - max_len
        cx = x[start:]
        cy = y[start:]
        first = 3 if start == 0 else min(int(y[start - 1]), 2)
        prev = _prev_limb_onehot(cy, first_label=first)
        chunks.append((np.concatenate([cx, prev], axis=1), cy))

    return chunks


def load_train_data_pair(files: list[Path]) -> list[tuple]:
    """Load (x, y, x_mirror, y_mirror) for each file without choosing mirror yet.

    Storing both versions in memory lets us re-randomize mirror choice each epoch
    with zero disk I/O overhead.
    """
    all_data = []
    for f in tqdm(files, desc='Loading training data'):
        d = np.load(f)
        all_data.append((
            d['x'].astype(np.float32),
            d['y'].astype(np.float32),
            d['x_mirror'].astype(np.float32),
            d['y_mirror'].astype(np.float32),
        ))
    return all_data


def build_both_epoch_chunks(train_data: list[tuple]) -> list:
    """Include BOTH original and mirror for every chart, with prev-limb feature.

    Using both versions forces the model to be L/R neutral and learn the
    alternation pattern from local context (prev-limb teacher forcing).
    """
    all_chunks = []
    for x, y, xm, ym in train_data:
        for chunk in make_chunks_with_prev(x, y):
            all_chunks.append(chunk)
        for chunk in make_chunks_with_prev(xm, ym):
            all_chunks.append(chunk)
    return all_chunks


def load_val_chunks(files: list[Path]) -> list[tuple]:
    all_chunks = []
    for f in tqdm(files, desc='Loading val data'):
        d = np.load(f)
        x = d['x'].astype(np.float32)
        y = d['y'].astype(np.float32)
        for chunk in make_chunks_with_prev(x, y):
            all_chunks.append(chunk)
    return all_chunks


def log_class_distribution(train_data: list[tuple]) -> None:
    counts = np.zeros(3, dtype=np.int64)
    for _, y, _, _ in train_data:
        valid = y[y >= 0].astype(int)
        for c in range(3):
            counts[c] += np.sum(valid == c)
    total = counts.sum()
    pct = counts / total * 100
    logger.info(f'Class dist (L/R/Either): {counts[0]} ({pct[0]:.1f}%) / {counts[1]} ({pct[1]:.1f}%) / {counts[2]} ({pct[2]:.1f}%)')


def build_batches(chunks, batch_size, shuffle=True, seed=0, with_impossible_pairs=False):
    # Sort by length to minimize padding waste within each batch
    sorted_idx = sorted(range(len(chunks)), key=lambda i: chunks[i][0].shape[0])
    batches = [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    for batch_idxs in batches:
        max_len = max(chunks[i][0].shape[0] for i in batch_idxs)
        batch_x, batch_y, batch_pad, batch_loss = [], [], [], []
        pb, pi_list, pj_list = [], [], []
        for b_idx, i in enumerate(batch_idxs):
            x, y = chunks[i]
            L = x.shape[0]
            x_padded = np.zeros((max_len, x.shape[1]), dtype=np.float32)
            y_padded = np.full((max_len,), -1, dtype=np.float32)
            x_padded[:L] = x
            y_padded[:L] = y
            # True = padding position (attention will ignore these)
            pad_mask = np.zeros(max_len, dtype=bool)
            pad_mask[L:] = True
            loss_mask = np.zeros(max_len, dtype=bool)
            loss_mask[:L] = True
            batch_x.append(x_padded)
            batch_y.append(y_padded)
            batch_pad.append(pad_mask)
            batch_loss.append(loss_mask)
            if with_impossible_pairs:
                for (ii, jj) in find_impossible_pairs(x):
                    pb.append(b_idx)
                    pi_list.append(ii)
                    pj_list.append(jj)
        batch = {
            'x': np.array(batch_x),
            'y': np.array(batch_y),
            'padding_mask': np.array(batch_pad),
            'loss_mask': np.array(batch_loss),
        }
        if with_impossible_pairs:
            batch['pairs_b'] = np.array(pb, dtype=np.int32)
            batch['pairs_i'] = np.array(pi_list, dtype=np.int32)
            batch['pairs_j'] = np.array(pj_list, dtype=np.int32)
        yield batch


def compute_loss(logits, y, loss_mask, smoothing=0.1, class_weights=None):
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

    if smoothing > 0:
        # Blend hard CE with uniform distribution to reduce overconfidence
        uniform_loss = -mx.mean(log_softmax, axis=-1)
        nll = (1.0 - smoothing) * nll + smoothing * uniform_loss

    if class_weights is not None:
        # Upweight rare classes (especially class 2 = "either")
        token_weights = mx.take(class_weights, mx.clip(y_flat, 0, C - 1))
        nll = nll * token_weights

    return mx.sum(nll * valid_mask) / valid_count


def compute_accuracy(logits, y, loss_mask):
    """Returns (n_correct, n_valid) for proper sample-weighted aggregation."""
    preds = mx.argmax(logits, axis=-1)
    target = y.astype(mx.int32)
    mask = (loss_mask & (y >= 0)).astype(mx.int32)
    correct = ((preds == target) & (mask == 1)).astype(mx.int32)
    return mx.sum(correct), mx.sum(mask)


def evaluate(model, val_chunks, batch_size):
    model.eval()
    total_loss, total_correct, total_tokens, n_batches = 0.0, 0, 0, 0
    class_correct = np.zeros(3, dtype=np.int64)
    class_total = np.zeros(3, dtype=np.int64)
    for batch in build_batches(val_chunks, batch_size, shuffle=False):
        x = mx.array(batch['x'])
        y = mx.array(batch['y'])
        pm = mx.array(batch['padding_mask'])
        lm = mx.array(batch['loss_mask'])
        logits = model(x, pm)
        loss = compute_loss(logits, y, lm)
        correct, tokens = compute_accuracy(logits, y, lm)
        mx.eval(logits, loss, correct, tokens)
        total_loss += float(loss)
        total_correct += int(correct)
        total_tokens += int(tokens)
        n_batches += 1
        # Per-class accuracy
        preds_np = np.array(mx.argmax(logits, axis=-1))
        y_np = np.array(y).astype(int)
        lm_np = np.array(lm)
        for c in range(3):
            mask = lm_np & (y_np == c)
            class_total[c] += mask.sum()
            class_correct[c] += (preds_np[mask] == c).sum()
    per_class = {c: class_correct[c] / max(class_total[c], 1) for c in range(3)}
    return {
        'loss': total_loss / max(n_batches, 1),
        'acc': total_correct / max(total_tokens, 1),
        'per_class': per_class,
        'class_total': class_total,
    }


def cosine_schedule(step, total, lr_max, warmup_steps):
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total - warmup_steps, 1)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_ss_to_batch(x_np: np.ndarray, logits_np: np.ndarray, ss_prob: float, rng) -> np.ndarray:
    """Scheduled sampling: replace teacher-forced prev_limb features with model predictions.

    For each token position i >= 1, with probability ss_prob, swap x[b,i,-4:] from
    ground-truth one-hot to argmax(logits[b,i-1]) one-hot.  Position 0 is always kept
    (it holds the START token, no previous prediction available).
    """
    B, L, D = x_np.shape
    if L <= 1 or ss_prob <= 0:
        return x_np
    # preds shape: (B, L-1) with values in {0, 1, 2}
    preds = np.argmax(logits_np[:, :-1, :], axis=-1)
    preds_oh = np.eye(4, dtype=np.float32)[np.clip(preds, 0, 2)]  # (B, L-1, 4)
    mask = (rng.random((B, L - 1)) < ss_prob)[:, :, None]         # (B, L-1, 1)
    x_ss = x_np.copy()
    x_ss[:, 1:, -4:] = np.where(mask, preds_oh, x_np[:, 1:, -4:])
    return x_ss


def evaluate_ar_accuracy(model: nn.Module, val_chunks: list, batch_size: int) -> float:
    """2-pass autoregressive accuracy: pass-1 teacher-forcing → pass-2 with those predictions.

    This approximates real inference accuracy (where no oracle prev_limb is available).
    Accurate when pass-1 predictions are mostly correct.
    """
    model.eval()
    total_correct, total_tokens = 0, 0
    for batch in build_batches(val_chunks, batch_size, shuffle=False):
        x = mx.array(batch['x'])
        y = mx.array(batch['y'])
        pm = mx.array(batch['padding_mask'])
        lm = mx.array(batch['loss_mask'])

        # Pass 1: teacher forcing
        logits1 = model(x, pm)
        mx.eval(logits1)
        preds1 = np.array(mx.argmax(logits1, axis=-1))  # (B, L)

        # Build autoregressive input: prev_limb[i] = one_hot(preds1[i-1]) for i >= 1
        x_np = np.array(x)
        B, L, D = x_np.shape
        if L > 1:
            preds_prev = np.clip(preds1[:, :-1], 0, 2)  # (B, L-1)
            prev_oh = np.eye(4, dtype=np.float32)[preds_prev]
            x_ar = x_np.copy()
            x_ar[:, 1:, -4:] = prev_oh
        else:
            x_ar = x_np

        # Pass 2: AR forward
        logits2 = model(mx.array(x_ar), pm)
        correct, tokens = compute_accuracy(logits2, y, lm)
        mx.eval(correct, tokens)
        total_correct += int(correct)
        total_tokens += int(tokens)

    return total_correct / max(total_tokens, 1)


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
    ss_prob_max: float = 0.0,
    large_model: bool = False,
    resume_from: str | None = None,
    impossible_penalty: float = 0.0,
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

    # Load both original and mirror into RAM; mirror is re-randomized each epoch
    train_data_pair = load_train_data_pair(train_files)
    val_chunks = load_val_chunks(val_files)
    logger.info(f"Train files: {len(train_files)}, Val chunks: {len(val_chunks)}")
    log_class_distribution(train_data_pair)
    class_weights_mx = mx.array(np.array([1.0, 1.0, class_weight_e], dtype=np.float32)) if class_weight_e != 1.0 else None

    input_dim = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM

    if limit_charts and limit_charts <= 200:
        logger.info("SMOKE TEST: Using smaller model.")
        model = LimbSequenceTransformer(
            input_dim=input_dim, d_model=64, n_heads=4, n_layers=2, ffn_dim=128,
        )
        d_model_save, n_heads_save, n_layers_save, ffn_dim_save = 64, 4, 2, 128
    elif large_model:
        logger.info("LARGE MODEL: d_model=384, n_layers=8")
        model = LimbSequenceTransformer(
            input_dim=input_dim, d_model=384, n_heads=8, n_layers=8, ffn_dim=1536,
        )
        d_model_save, n_heads_save, n_layers_save, ffn_dim_save = 384, 8, 8, 1536
    else:
        model = LimbSequenceTransformer(
            input_dim=input_dim, d_model=256, n_heads=8, n_layers=6, ffn_dim=1024,
        )
        d_model_save, n_heads_save, n_layers_save, ffn_dim_save = 256, 8, 6, 1024

    if resume_from and os.path.exists(resume_from):
        model.load_weights(resume_from)
        mx.eval(model.parameters())
        logger.info(f"Resumed model from {resume_from}")
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    logger.info(f"Model: {n_params:,} parameters")

    # Estimate steps_per_epoch (both mirror versions, so 2× the file count)
    dummy_chunks = build_both_epoch_chunks(train_data_pair)
    steps_per_epoch = max(len(dummy_chunks) // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    del dummy_chunks
    logger.info(f"~{steps_per_epoch} steps/epoch, {total_steps} total, {warmup_steps} warmup steps")

    # LR schedule: linear warmup → cosine decay, managed inside the compiled graph
    lr_sched = join_schedules(
        [linear_schedule(0.0, lr, warmup_steps),
         cosine_decay(lr, max(total_steps - warmup_steps, 1), end=1e-6)],
        [warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_sched, weight_decay=0.05)
    state = [model.state, optimizer.state]

    use_impossible_penalty = impossible_penalty > 0.0
    impossible_penalty_mx = mx.array(impossible_penalty) if use_impossible_penalty else None

    def loss_fn(mdl, x, y, pm, lm, pb=None, pi=None, pj=None):
        logits = mdl(x, pm)
        loss = compute_loss(logits, y, lm,
                            smoothing=label_smoothing,
                            class_weights=class_weights_mx)
        if pb is not None and pb.shape[0] > 0:
            loss = loss + impossible_penalty_mx * compute_impossible_penalty(logits, pb, pi, pj)
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Step (teacher forcing only) — compile disabled to avoid recompile stalls
    def step(x, y, pm, lm, pb=None, pi=None, pj=None):
        loss, grads = loss_and_grad_fn(model, x, y, pm, lm, pb, pi, pj)
        grads, _ = mx_clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss

    # Step with scheduled sampling — compile disabled
    def step_ss(x, y, pm, lm, ss_prob_mx, pb=None, pi=None, pj=None):
        B, L, _ = x.shape
        # Pass 1: inference for SS predictions (no explicit stop_gradient needed
        # since loss_and_grad_fn only differentiates through pass-2's model call)
        logits_tf = model(x, pm)  # (B, L, 3)
        preds = mx.argmax(logits_tf[:, :-1, :], axis=-1)  # (B, L-1)
        # One-hot over 4 dims (L/R/E/START); preds in 0..2, never produces dim 3
        preds_oh = (preds[:, :, None] == mx.arange(4)[None, None, :]).astype(x.dtype)
        mix_mask = (mx.random.uniform(shape=(B, L-1, 1)) < ss_prob_mx).astype(x.dtype)
        mixed_prev = mix_mask * preds_oh + (1.0 - mix_mask) * x[:, 1:, -4:]
        # Build SS input; stop_gradient so grads don't flow back through argmax
        x_ss = mx.stop_gradient(mx.concatenate([
            x[:, :1, :],
            mx.concatenate([x[:, 1:, :-4], mixed_prev], axis=-1),
        ], axis=1))
        # Pass 2: loss + grad through model(x_ss, pm) only
        loss, grads = loss_and_grad_fn(model, x_ss, y, pm, lm, pb, pi, pj)
        grads, _ = mx_clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss

    best_acc = 0.0
    no_improve = 0
    global_step = 0

    for epoch in range(epochs):
        logger.info(f'=== Epoch {epoch+1}/{epochs} starting ===')

        # Scheduled-sampling probability: 0 during warmup, then ramp linearly
        if ss_prob_max > 0 and epoch >= warmup_epochs:
            ramp = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            ss_prob = min(ss_prob_max * ramp, ss_prob_max)
        else:
            ss_prob = 0.0

        # Both original and mirror for every chart (forces L/R symmetry)
        train_chunks = build_both_epoch_chunks(train_data_pair)
        logger.info(
            f'  {len(train_chunks)} chunks ({len(train_data_pair)} charts × 2 versions)'
            + (f', ss_prob={ss_prob:.3f}' if ss_prob > 0 else '')
        )

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(
            build_batches(train_chunks, batch_size, shuffle=True, seed=seed + epoch,
                          with_impossible_pairs=use_impossible_penalty),
            desc='Training',
            leave=False,
        ):
            x = mx.array(batch['x'])
            y = mx.array(batch['y'])
            pm = mx.array(batch['padding_mask'])
            lm = mx.array(batch['loss_mask'])

            pb = mx.array(batch['pairs_b']) if use_impossible_penalty else None
            pi = mx.array(batch['pairs_i']) if use_impossible_penalty else None
            pj = mx.array(batch['pairs_j']) if use_impossible_penalty else None

            if ss_prob > 0:
                loss = step_ss(x, y, pm, lm, mx.array(ss_prob), pb, pi, pj)
            else:
                loss = step(x, y, pm, lm, pb, pi, pj)
            if global_step % 8 == 0:
                mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(loss)
            if global_step % 50 == 0:
                lr_now = float(lr_sched(global_step))
                logger.info(f'  step {global_step}, loss={float(loss):.4f}, lr={lr_now:.2e}')
            n_batches += 1
            global_step += 1

        mx.eval(model.parameters(), optimizer.state)
        train_loss = epoch_loss / max(n_batches, 1)
        lr_now = float(lr_sched(global_step))

        val_metrics = evaluate(model, val_chunks, batch_size)
        pc = val_metrics['per_class']
        ct = val_metrics['class_total']
        logger.info(
            f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']*100:.1f}%, "
            f"L={pc[0]*100:.1f}%(n={ct[0]}) R={pc[1]*100:.1f}%(n={ct[1]}) E={pc[2]*100:.1f}%(n={ct[2]}), "
            f"lr={lr_now:.2e}"
        )

        # Autoregressive accuracy (2-pass approximation; no oracle prev_limb)
        ar_acc = evaluate_ar_accuracy(model, val_chunks, batch_size)
        model.train()
        gap = val_metrics['acc'] - ar_acc
        logger.info(f"  AR val_acc={ar_acc*100:.1f}%  (oracle gap={gap*100:.1f}pp)")

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            no_improve = 0
            save_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
            save_model(save_path, model)
            with open(os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.meta'), 'w') as f:
                json.dump({
                    'input_dim': input_dim,
                    'd_model': d_model_save,
                    'n_heads': n_heads_save,
                    'n_layers': n_layers_save,
                    'ffn_dim': ffn_dim_save,
                    'n_classes': 3,
                }, f)
            logger.success(f"New best saved: val_acc={best_acc*100:.1f}%")
        else:
            no_improve += 1
            logger.info(f"  No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    final_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-final.safetensors')
    save_model(final_path, model)
    logger.success(f"Training complete. Best val_acc: {best_acc*100:.1f}%")

    config = {
        'sd': sd, 'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
        'seed': seed, 'input_dim': input_dim, 'best_acc': best_acc,
        'd_model': d_model_save, 'n_heads': n_heads_save,
        'n_layers': n_layers_save, 'ffn_dim': ffn_dim_save,
    }
    with open(os.path.join(out_dir, 'train_config.json'), 'w') as f:
        json.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--singles_or_doubles', type=str, required=True)
    parser.add_argument('--manual_chart_struct_folder', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--limit_charts', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--class_weight_e', type=float, default=1.0)
    parser.add_argument('--ss_prob_max', type=float, default=0.0,
                        help='Max scheduled-sampling prob (linearly ramped post-warmup). 0=off.')
    parser.add_argument('--large_model', action='store_true',
                        help='Use d_model=384, n_layers=8 (~14M params) instead of default 5M.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to .safetensors checkpoint to resume training from.')
    parser.add_argument('--impossible_penalty', type=float, default=0.0,
                        help='Weight for impossible same-foot bracket penalty. 0=off. '
                             'Try 10.0–50.0 for a penalty fine-tuning epoch.')
    args = parser.parse_args()

    train(
        folder=args.manual_chart_struct_folder,
        sd=args.singles_or_doubles,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        limit_charts=args.limit_charts,
        label_smoothing=args.label_smoothing,
        class_weight_e=args.class_weight_e,
        ss_prob_max=args.ss_prob_max,
        large_model=args.large_model,
        resume_from=args.resume_from,
        impossible_penalty=args.impossible_penalty,
    )
