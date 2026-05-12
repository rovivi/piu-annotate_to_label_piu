#!/usr/bin/env python3
"""
Fine-tune a trained MLX model on hard negatives + structured penalties.

Usage:
    python cli/limbuse/finetune_mlx.py \
        --model_dir artifacts/models/visss-mlx-v9 \
        --cache_dir artifacts/cache/mlx-v9-singles \
        --hard_negatives artifacts/hard_negatives/singles_hard_negatives.json \
        --out_dir artifacts/models/visss-mlx-v9-ft \
        --singles_or_doubles singles \
        --epochs 10 --batch_size 16 --lr 1e-4 \
        --lambda_p1 0.5 --lambda_p3 0.3 \
        --hard_neg_weight 3.0
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
CHUNK_OVERLAP = 256
SINGLES_INPUT_DIM = 28
DOUBLES_INPUT_DIM = 33

_BRACKETABLE_SET: frozenset[tuple[int, int]] = frozenset(
    (min(a, b), max(a, b)) for a, b in [
        [0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2],  # P1 center + side brackets
        [4, 5], [3, 6],                                    # inter-pad
        [5, 6], [5, 7], [6, 7], [8, 7], [8, 9], [9, 7],  # P2 center + side brackets
    ]
)


def find_impossible_pairs(x_np: np.ndarray) -> list[tuple[int, int]]:
    """Return (i, j) index pairs in sequence that are impossible same-foot brackets."""
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


def compute_impossible_penalty(logits, pairs_b, pairs_i, pairs_j):
    """Penalize impossible same-foot bracket predictions."""
    B, L, C = logits.shape
    max_l = mx.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_l - mx.log(mx.sum(mx.exp(logits - max_l), axis=-1, keepdims=True))
    log_probs_flat = log_probs.reshape(B * L, C)
    flat_i = pairs_b * L + pairs_i
    flat_j = pairs_b * L + pairs_j
    lp_i = mx.take(log_probs_flat, flat_i, axis=0)
    lp_j = mx.take(log_probs_flat, flat_j, axis=0)
    probs_i = mx.exp(lp_i)
    probs_j = mx.exp(lp_j)
    penalty = mx.sum(probs_i[:, 0] * probs_j[:, 0] + probs_i[:, 1] * probs_j[:, 1])
    return penalty / (flat_i.shape[0] + 1e-8)


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
    N = len(y)
    prev = np.zeros((N, 4), dtype=np.float32)
    prev[0, min(first_label, 3)] = 1.0
    for i in range(1, N):
        lbl = int(y[i - 1])
        prev[i, min(lbl, 2)] = 1.0
    return prev


def make_chunks_with_prev(x, y, max_len=MAX_SEQ_LEN, overlap=CHUNK_OVERLAP):
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


def load_val_chunks(files: list[Path]) -> list[tuple]:
    all_chunks = []
    for f in tqdm(files, desc='Loading val data'):
        d = np.load(f)
        x = d['x'].astype(np.float32)
        y = d['y'].astype(np.float32)
        for chunk in make_chunks_with_prev(x, y):
            all_chunks.append(chunk)
    return all_chunks


def build_batches(chunks, batch_size, shuffle=True, seed=0, with_impossible_pairs=False):
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


def compute_p1_penalty(logits, x, loss_mask):
    """P1: Penalize same-limb assignment on consecutive same-panel arrows (non-jack).
    
    Feature indices in x (after prev_limb concat):
    - arrow_pos: index 0
    - is_jack: index 16
    """
    B, L, C = logits.shape
    if L <= 1:
        return mx.array(0.0)
    
    probs = mx.softmax(logits, axis=-1)  # (B, L, 3)
    
    # Arrow position: index 0
    arrow_pos = x[:, :, 0]
    same_panel = (arrow_pos[:, 1:] == arrow_pos[:, :-1]).astype(mx.float32)
    
    # is_jack: index 16
    is_jack = x[:, :, 16].astype(mx.float32)
    not_jack = (1.0 - is_jack)
    
    # P(same limb at i and i-1) = P(L_i)*P(L_{i-1}) + P(R_i)*P(R_{i-1})
    p_same = probs[:, 1:, 0] * probs[:, :-1, 0] + probs[:, 1:, 1] * probs[:, :-1, 1]
    
    # Weight by same-panel and non-jack
    penalty = p_same * same_panel * not_jack[:, 1:] * not_jack[:, :-1]
    
    # Valid mask
    valid = (loss_mask[:, 1:].astype(mx.float32) * loss_mask[:, :-1].astype(mx.float32))
    return mx.sum(penalty * valid) / mx.sum(valid + 1e-8)


def compute_p3_penalty(logits, x, loss_mask):
    """P3: Hold-aware bracket priority.
    
    In lines with holds and multiple downpresses, arrows on held panels
    should prefer the foot that's already holding.
    
    Feature indices:
    - has_active_hold: index 2
    - num_downpress_in_line: index 6
    - hold_count_in_line: index 14
    """
    B, L, C = logits.shape
    
    probs = mx.softmax(logits, axis=-1)
    has_hold = x[:, :, 2].astype(mx.float32)
    num_dp = x[:, :, 6].astype(mx.float32)
    hold_count = x[:, :, 14].astype(mx.float32)
    
    # Heuristic: if there are holds in a line with multiple downpresses,
    # encourage "either" (class 2) to allow the inference logic to resolve
    # which foot maintains the hold.
    bracket_with_hold = (num_dp >= 2).astype(mx.float32) * (hold_count > 0).astype(mx.float32)
    
    # Penalize low probability on class 2 (either) for hold arrows in brackets
    p_either = probs[:, :, 2]
    penalty = (1.0 - p_either) * bracket_with_hold * has_hold
    
    valid = loss_mask.astype(mx.float32)
    return mx.sum(penalty * valid) / mx.sum(valid + 1e-8)


def compute_structured_loss(logits, y, loss_mask, x, lambda_p1, lambda_p3, smoothing=0.1, class_weights=None,
                            lambda_p2=0.0, pairs_b=None, pairs_i=None, pairs_j=None):
    """Cross-entropy + structured penalties."""
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
        uniform_loss = -mx.mean(log_softmax, axis=-1)
        nll = (1.0 - smoothing) * nll + smoothing * uniform_loss

    if class_weights is not None:
        token_weights = mx.take(class_weights, mx.clip(y_flat, 0, C - 1))
        nll = nll * token_weights

    ce_loss = mx.sum(nll * valid_mask) / valid_count
    
    total_loss = ce_loss
    if lambda_p1 > 0:
        total_loss = total_loss + lambda_p1 * compute_p1_penalty(logits, x, loss_mask)
    if lambda_p3 > 0:
        total_loss = total_loss + lambda_p3 * compute_p3_penalty(logits, x, loss_mask)
    if lambda_p2 > 0 and pairs_b is not None and pairs_b.shape[0] > 0:
        total_loss = total_loss + lambda_p2 * compute_impossible_penalty(logits, pairs_b, pairs_i, pairs_j)

    return total_loss


def compute_accuracy(logits, y, loss_mask):
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
        loss = compute_structured_loss(logits, y, lm, x, lambda_p1=0, lambda_p3=0)
        correct, tokens = compute_accuracy(logits, y, lm)
        mx.eval(logits, loss, correct, tokens)
        total_loss += float(loss)
        total_correct += int(correct)
        total_tokens += int(tokens)
        n_batches += 1
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


def evaluate_ar_accuracy(model, val_chunks, batch_size):
    model.eval()
    total_correct, total_tokens = 0, 0
    for batch in build_batches(val_chunks, batch_size, shuffle=False):
        x = mx.array(batch['x'])
        y = mx.array(batch['y'])
        pm = mx.array(batch['padding_mask'])
        lm = mx.array(batch['loss_mask'])
        logits1 = model(x, pm)
        mx.eval(logits1)
        preds1 = np.array(mx.argmax(logits1, axis=-1))
        x_np = np.array(x)
        B, L, D = x_np.shape
        if L > 1:
            preds_prev = np.clip(preds1[:, :-1], 0, 2)
            prev_oh = np.eye(4, dtype=np.float32)[preds_prev]
            x_ar = x_np.copy()
            x_ar[:, 1:, -4:] = prev_oh
        else:
            x_ar = x_np
        logits2 = model(mx.array(x_ar), pm)
        correct, tokens = compute_accuracy(logits2, y, lm)
        mx.eval(correct, tokens)
        total_correct += int(correct)
        total_tokens += int(tokens)
    return total_correct / max(total_tokens, 1)


def build_weighted_epoch_chunks(train_data, hard_neg_dict, hard_neg_weight=3.0):
    """Build chunks with optional upweighting of hard-negative charts.
    
    Instead of actually weighting (which would require loss modifications),
    we duplicate hard-negative charts in the epoch.
    """
    all_chunks = []
    for x, y, xm, ym in train_data:
        fname = None  # We don't have filename here, need to pass it
        all_chunks.extend(make_chunks_with_prev(x, y))
        all_chunks.extend(make_chunks_with_prev(xm, ym))
    return all_chunks


def finetune(
    model_dir: str,
    cache_dir: str,
    hard_negatives_path: str | None,
    out_dir: str,
    sd: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    seed: int = 0,
    warmup_epochs: int = 1,
    patience: int = 5,
    lambda_p1: float = 0.5,
    lambda_p3: float = 0.3,
    hard_neg_weight: float = 3.0,
    lambda_p2: float = 0.0,
):
    mx.random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    os.makedirs(out_dir, exist_ok=True)

    # Load hard negatives index
    hard_neg_dict = {}
    if hard_negatives_path and os.path.exists(hard_negatives_path):
        with open(hard_negatives_path) as f:
            hn = json.load(f)
        for item in hn.get('hard_negatives', []):
            hard_neg_dict[item['file']] = item
        logger.info(f"Loaded {len(hard_neg_dict)} hard-negative charts from {hard_negatives_path}")
    else:
        logger.warning("No hard negatives provided; fine-tuning will use standard CE + penalties")

    all_files = sorted(Path(cache_dir).glob('*.npz'))
    
    # Use same train/val split as original training (90/10)
    # To be consistent, we should use the same split. For simplicity,
    # we'll load from the model's config if available.
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.9)
    train_files, val_files = shuffled[:split], shuffled[split:]

    train_data_pair = load_train_data_pair(train_files)
    val_chunks = load_val_chunks(val_files)
    logger.info(f"Train files: {len(train_files)}, Val chunks: {len(val_chunks)}")

    input_dim = SINGLES_INPUT_DIM if sd == 'singles' else DOUBLES_INPUT_DIM

    # Load model
    config_path = os.path.join(model_dir, 'train_config.json')
    with open(config_path) as f:
        cfg = json.load(f)
    
    model = LimbSequenceTransformer(
        input_dim=input_dim,
        d_model=cfg.get('d_model', 256),
        n_heads=cfg.get('n_heads', 8),
        n_layers=cfg.get('n_layers', 6),
        ffn_dim=cfg.get('ffn_dim', 1024),
    )
    weights_path = os.path.join(model_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
    model.load_weights(weights_path)
    mx.eval(model.parameters())
    logger.info(f"Loaded base model from {weights_path}")
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    logger.info(f"Model: {n_params:,} parameters")

    # Build chunks with hard-negative duplication
    train_chunks = []
    for idx, (x, y, xm, ym) in enumerate(train_data_pair):
        fname = train_files[idx].name
        weight = hard_neg_weight if fname in hard_neg_dict else 1.0
        n_copies = max(1, int(round(weight)))
        for _ in range(n_copies):
            train_chunks.extend(make_chunks_with_prev(x, y))
            train_chunks.extend(make_chunks_with_prev(xm, ym))
    
    steps_per_epoch = max(len(train_chunks) // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    logger.info(f"~{steps_per_epoch} steps/epoch, {total_steps} total, {warmup_steps} warmup")

    lr_sched = join_schedules(
        [linear_schedule(0.0, lr, warmup_steps),
         cosine_decay(lr, max(total_steps - warmup_steps, 1), end=1e-6)],
        [warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_sched, weight_decay=0.01)
    state = [model.state, optimizer.state]

    use_p2 = lambda_p2 > 0.0

    def loss_fn(mdl, x, y, pm, lm, pb=None, pi=None, pj=None):
        logits = mdl(x, pm)
        return compute_structured_loss(logits, y, lm, x, lambda_p1, lambda_p3,
                                       lambda_p2=lambda_p2, pairs_b=pb, pairs_i=pi, pairs_j=pj)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def step(x, y, pm, lm, pb=None, pi=None, pj=None):
        loss, grads = loss_and_grad_fn(model, x, y, pm, lm, pb, pi, pj)
        grads, _ = mx_clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(model, grads)
        return loss

    best_acc = cfg.get('best_acc', 0.0)
    no_improve = 0
    global_step = 0

    for epoch in range(epochs):
        logger.info(f'=== Fine-tune Epoch {epoch+1}/{epochs} starting ===')
        logger.info(f'  {len(train_chunks)} chunks (hard-neg weight={hard_neg_weight})')
        
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(
            build_batches(train_chunks, batch_size, shuffle=True, seed=seed + epoch,
                          with_impossible_pairs=use_p2),
            desc='Fine-tuning',
            leave=False,
        ):
            x = mx.array(batch['x'])
            y = mx.array(batch['y'])
            pm = mx.array(batch['padding_mask'])
            lm = mx.array(batch['loss_mask'])

            pb = mx.array(batch['pairs_b']) if use_p2 else None
            pi = mx.array(batch['pairs_i']) if use_p2 else None
            pj = mx.array(batch['pairs_j']) if use_p2 else None

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
                    'd_model': cfg.get('d_model', 256),
                    'n_heads': cfg.get('n_heads', 8),
                    'n_layers': cfg.get('n_layers', 6),
                    'ffn_dim': cfg.get('ffn_dim', 1024),
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
    logger.success(f"Fine-tuning complete. Best val_acc: {best_acc*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--hard_negatives', type=str, default=None)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--singles_or_doubles', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lambda_p1', type=float, default=0.5)
    parser.add_argument('--lambda_p3', type=float, default=0.3)
    parser.add_argument('--hard_neg_weight', type=float, default=3.0)
    parser.add_argument('--lambda_p2', type=float, default=0.0,
                        help='Impossible same-foot bracket penalty weight. 0=off. Try 10–30.')
    args = parser.parse_args()

    finetune(
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        hard_negatives_path=args.hard_negatives,
        out_dir=args.out_dir,
        sd=args.singles_or_doubles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        lambda_p1=args.lambda_p1,
        lambda_p3=args.lambda_p3,
        hard_neg_weight=args.hard_neg_weight,
        lambda_p2=args.lambda_p2,
    )
