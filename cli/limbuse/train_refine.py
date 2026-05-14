#!/usr/bin/env python3
"""Train the refine (a.k.a. ``arrowlimbs_to_limb``) stage of the two-pass
limb predictor. Backend-agnostic — pass ``--backend mlx`` (default) or
``--backend torch``.

Pipeline
--------
    raw_features (D)                                        ┐
        + coarse_softmax (3, from frozen coarse model)       │ refine input
        + prev_limb one-hot (4, teacher forced from y)       │  (D + 7)
                                                            ┘
        ↓
    refine LimbSequenceTransformer (same architecture as coarse but with
    input_dim = D + 3 instead of D)
        ↓
    cross-entropy vs ground-truth limb labels (3-class)

The coarse model is loaded from a previously trained safetensors file and
**not updated**. Only the refine model receives gradients.

Output safetensors / meta files use the suffix ``arrowlimbs_to_limb`` so
``ModelSuite`` automatically picks them up and wraps them with
``RefineModel`` at inference time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from piu_annotate.ml.seq_data import (
    build_batches, build_both_epoch_chunks, list_cached_files,
    load_train_data_pair, load_val_chunks, train_val_split,
)


SINGLES_INPUT_DIM_RAW = 24       # base arrow features
SINGLES_INPUT_DIM = 28           # base + prev_limb (4) — coarse model input
DOUBLES_INPUT_DIM_RAW = 29
DOUBLES_INPUT_DIM = 33

REFINE_EXTRA_DIM = 3             # coarse softmax dims appended to base


# ---------------------------------------------------------------------------
# Loss math (numpy reference for both backends).
# ---------------------------------------------------------------------------


def _per_token_smoothed_ce(log_softmax, y_long, smoothing: float):
    """Returns per-token NLL with optional label smoothing.

    Implemented in framework-agnostic shape ops in the caller; this is just
    the formula reminder. Each backend implements the actual loss with its
    own native ops to keep grads flowing."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# MLX path
# ---------------------------------------------------------------------------


def _train_mlx(
    folder, sd, out_dir, coarse_path, coarse_meta_path,
    epochs, batch_size, lr, seed, warmup_epochs, patience,
    limit_charts, label_smoothing, large_model, resume_from,
):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.optimizers import (
        linear_schedule, cosine_decay, join_schedules,
        clip_grad_norm as mx_clip_grad_norm,
    )
    from mlx.utils import tree_flatten
    from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer

    mx.random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    files = list_cached_files(folder, limit=limit_charts)
    train_files, val_files = train_val_split(files, frac=0.9, seed=seed)
    train_data = load_train_data_pair(train_files)
    val_chunks = load_val_chunks(val_files)
    logger.info(f'[mlx] train={len(train_files)} val_chunks={len(val_chunks)}')

    # Coarse: load from .meta + safetensors, freeze.
    with open(coarse_meta_path) as f:
        cmeta = json.load(f)
    # ``input_dim`` in .meta is the full transformer dim (already includes prev_limb)
    coarse = LimbSequenceTransformer(
        input_dim=cmeta['input_dim'],
        d_model=cmeta.get('d_model', 256),
        n_heads=cmeta.get('n_heads', 8),
        n_layers=cmeta.get('n_layers', 6),
        ffn_dim=cmeta.get('ffn_dim', 1024),
        n_classes=cmeta.get('n_classes', 3),
    )
    coarse.load_weights(coarse_path)
    mx.eval(coarse.parameters())
    coarse.eval()

    raw_dim = SINGLES_INPUT_DIM_RAW if sd == 'singles' else DOUBLES_INPUT_DIM_RAW
    refine_input_dim = raw_dim + REFINE_EXTRA_DIM + 4  # +4 prev_limb

    if large_model:
        d_model, n_heads, n_layers, ffn_dim = 384, 8, 8, 1536
    else:
        d_model, n_heads, n_layers, ffn_dim = 256, 8, 6, 1024

    refine = LimbSequenceTransformer(
        input_dim=refine_input_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
    )
    if resume_from and os.path.exists(resume_from):
        refine.load_weights(resume_from)
    mx.eval(refine.parameters())
    n_params = sum(v.size for _, v in tree_flatten(refine.parameters()))
    logger.info(f'[mlx] refine params: {n_params:,}')

    dummy_chunks = build_both_epoch_chunks(train_data)
    steps_per_epoch = max(len(dummy_chunks) // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    del dummy_chunks

    lr_sched = join_schedules(
        [linear_schedule(0.0, lr, warmup_steps),
         cosine_decay(lr, max(total_steps - warmup_steps, 1), end=1e-6)],
        [warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=lr_sched, weight_decay=0.05)

    def _ce_smoothed(logits, y, lm, smoothing):
        B, L, C = logits.shape
        f = logits.reshape(B * L, C)
        yflat = y.reshape(B * L).astype(mx.int32)
        valid = (lm & (y >= 0)).astype(logits.dtype).reshape(B * L)
        valid_count = mx.sum(valid) + 1e-8
        m = mx.max(f, axis=-1, keepdims=True)
        log_softmax = (f - m) - mx.log(mx.sum(mx.exp(f - m), axis=-1, keepdims=True))
        nll = -mx.take_along_axis(log_softmax, yflat[:, None], axis=1).squeeze(-1)
        if smoothing > 0:
            uniform = -mx.mean(log_softmax, axis=-1)
            nll = (1.0 - smoothing) * nll + smoothing * uniform
        return mx.sum(nll * valid) / valid_count

    def loss_fn(refine_model, x_coarse, x_refine, y, pm, lm):
        # Forward refine only — gradients flow only here.
        logits = refine_model(x_refine, pm)
        return _ce_smoothed(logits, y, lm, label_smoothing)

    loss_and_grad_fn = nn.value_and_grad(refine, loss_fn)

    def step(x_coarse, x_refine, y, pm, lm):
        loss, grads = loss_and_grad_fn(refine, x_coarse, x_refine, y, pm, lm)
        grads, _ = mx_clip_grad_norm(grads, max_norm=1.0)
        optimizer.update(refine, grads)
        return loss

    best_acc, no_improve, gstep = 0.0, 0, 0
    for epoch in range(epochs):
        logger.info(f'=== [mlx] Epoch {epoch+1}/{epochs} ===')
        train_chunks = build_both_epoch_chunks(train_data)
        refine.train()
        epoch_loss, nb = 0.0, 0
        for batch in tqdm(
            build_batches(train_chunks, batch_size, shuffle=True, seed=seed + epoch),
            desc='Training refine', leave=False,
        ):
            x_full = mx.array(batch['x'])     # (B, L, D+4)
            y = mx.array(batch['y'])
            pm = mx.array(batch['padding_mask'])
            lm = mx.array(batch['loss_mask'])

            coarse_logits = coarse(x_full, pm)
            coarse_soft = mx.softmax(coarse_logits, axis=-1)         # (B, L, 3)
            coarse_soft = mx.stop_gradient(coarse_soft)
            # x_refine = [raw (D), prev_limb (4), coarse_soft (3)]
            # The cache stores x_full = [raw, prev_limb]. Insert coarse_soft
            # between raw and prev_limb so the refine model's feature order
            # is [raw, coarse_soft, prev_limb] — matches refine_input_dim.
            raw = x_full[:, :, :-4]
            prev = x_full[:, :, -4:]
            x_refine = mx.concatenate([raw, coarse_soft, prev], axis=-1)
            x_refine = mx.stop_gradient(x_refine)

            loss = step(x_full, x_refine, y, pm, lm)
            if gstep % 8 == 0:
                mx.eval(refine.parameters(), optimizer.state)
            epoch_loss += float(loss)
            if gstep % 50 == 0:
                lr_now = float(lr_sched(gstep))
                logger.info(f'  step {gstep}, loss={float(loss):.4f}, lr={lr_now:.2e}')
            nb += 1
            gstep += 1
        mx.eval(refine.parameters(), optimizer.state)
        train_loss = epoch_loss / max(nb, 1)

        # Eval
        refine.eval()
        total_correct, total_tokens = 0, 0
        for batch in build_batches(val_chunks, batch_size, shuffle=False):
            x_full = mx.array(batch['x'])
            y = mx.array(batch['y'])
            pm = mx.array(batch['padding_mask'])
            lm = mx.array(batch['loss_mask'])
            coarse_soft = mx.softmax(coarse(x_full, pm), axis=-1)
            raw = x_full[:, :, :-4]
            prev = x_full[:, :, -4:]
            x_refine = mx.concatenate([raw, coarse_soft, prev], axis=-1)
            logits = refine(x_refine, pm)
            preds = mx.argmax(logits, axis=-1)
            mask = (lm & (y >= 0)).astype(mx.int32)
            correct = ((preds == y.astype(mx.int32)) & (mask == 1)).astype(mx.int32)
            mx.eval(correct, mask)
            total_correct += int(mx.sum(correct))
            total_tokens += int(mx.sum(mask))
        val_acc = total_correct / max(total_tokens, 1)
        logger.info(f'Epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={val_acc*100:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            best_path = os.path.join(out_dir, f'{sd}-arrowlimbs_to_limb-mlx-best.safetensors')
            mx.save_safetensors(best_path, dict(tree_flatten(refine.parameters())))
            with open(os.path.join(out_dir, f'{sd}-arrowlimbs_to_limb-mlx-best.meta'), 'w') as f:
                json.dump({
                    'input_dim': raw_dim + REFINE_EXTRA_DIM + 4,  # raw + coarse_soft + prev_limb
                    'd_model': d_model, 'n_heads': n_heads,
                    'n_layers': n_layers, 'ffn_dim': ffn_dim,
                    'n_classes': 3, 'is_refine': True,
                    'coarse_weights': os.path.basename(coarse_path),
                }, f)
            logger.success(f'[mlx] New best: {best_acc*100:.2f}%')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f'[mlx] Early stop at epoch {epoch+1}')
                break

    logger.success(f'[mlx] Refine training complete. Best={best_acc*100:.2f}%')


# ---------------------------------------------------------------------------
# Torch path
# ---------------------------------------------------------------------------


def _train_torch(
    folder, sd, out_dir, coarse_path, coarse_meta_path,
    epochs, batch_size, lr, seed, warmup_epochs, patience,
    limit_charts, label_smoothing, large_model, resume_from, device,
    coarse_init: bool = False,
):
    import torch
    from piu_annotate.ml.arch_torch import (
        build_model, pick_device, save_safetensors, load_safetensors,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    dev = pick_device(prefer=device)
    logger.info(f'[torch] Device: {dev}')

    files = list_cached_files(folder, limit=limit_charts)
    train_files, val_files = train_val_split(files, frac=0.9, seed=seed)
    train_data = load_train_data_pair(train_files)
    val_chunks = load_val_chunks(val_files)

    with open(coarse_meta_path) as f:
        cmeta = json.load(f)
    # ``input_dim`` in .meta is the full transformer dim (already includes prev_limb)
    coarse = build_model(
        input_dim=cmeta['input_dim'],
        d_model=cmeta.get('d_model', 256),
        n_heads=cmeta.get('n_heads', 8),
        n_layers=cmeta.get('n_layers', 6),
        ffn_dim=cmeta.get('ffn_dim', 1024),
    ).to(dev)
    load_safetensors(coarse, coarse_path, device=dev)
    coarse.eval()
    for p in coarse.parameters():
        p.requires_grad = False

    raw_dim = SINGLES_INPUT_DIM_RAW if sd == 'singles' else DOUBLES_INPUT_DIM_RAW
    refine_input_dim = raw_dim + REFINE_EXTRA_DIM + 4
    if large_model:
        d_model, n_heads, n_layers, ffn_dim = 384, 8, 8, 1536
    else:
        d_model, n_heads, n_layers, ffn_dim = 256, 8, 6, 1024
    refine = build_model(
        input_dim=refine_input_dim, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
    ).to(dev)
    if coarse_init:
        import safetensors.torch as _st
        coarse_state = _st.load_file(coarse_path, device='cpu')
        refine_state = refine.state_dict()
        copied, skipped = 0, 0
        for k, v in coarse_state.items():
            if k not in refine_state:
                skipped += 1
                continue
            rs = refine_state[k]
            if v.shape == rs.shape:
                refine_state[k] = v.to(rs.device)
                copied += 1
            elif k == 'input_proj.weight':
                # coarse input: [raw(D), prev_limb(4)] → shape (d_model, D+4)
                # refine input: [raw(D), coarse_soft(3), prev_limb(4)] → (d_model, D+7)
                new_w = rs.clone()
                new_w[:, :raw_dim] = v[:, :raw_dim]    # raw feature cols
                new_w[:, -4:] = v[:, -4:]              # prev_limb cols; coarse_soft stays at init
                refine_state[k] = new_w
                copied += 1
        refine.load_state_dict(refine_state)
        logger.info(f'[torch] coarse-init: {copied} tensors copied, {skipped} skipped (shape/key mismatch)')
    elif resume_from and os.path.exists(resume_from):
        load_safetensors(refine, resume_from, device=dev)
    logger.info(f'[torch] refine params: {sum(p.numel() for p in refine.parameters()):,}')

    dummy_chunks = build_both_epoch_chunks(train_data)
    steps_per_epoch = max(len(dummy_chunks) // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    del dummy_chunks

    optimizer = torch.optim.AdamW(refine.parameters(), lr=lr, weight_decay=0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _ce_smoothed(logits, y, lm, smoothing):
        B, L, C = logits.shape
        f = logits.reshape(B * L, C)
        yflat = y.reshape(B * L).clamp_min(0).long()
        valid = (lm & (y >= 0)).reshape(B * L).to(logits.dtype)
        log_softmax = torch.nn.functional.log_softmax(f, dim=-1)
        nll = -log_softmax.gather(1, yflat.unsqueeze(-1)).squeeze(-1)
        if smoothing > 0:
            uniform = -log_softmax.mean(dim=-1)
            nll = (1.0 - smoothing) * nll + smoothing * uniform
        return (nll * valid).sum() / valid.sum().clamp_min(1e-8)

    best_acc, no_improve, gstep = 0.0, 0, 0
    for epoch in range(epochs):
        logger.info(f'=== [torch] Epoch {epoch+1}/{epochs} ===')
        train_chunks = build_both_epoch_chunks(train_data)
        refine.train()
        epoch_loss, nb = 0.0, 0
        for batch in tqdm(
            build_batches(train_chunks, batch_size, shuffle=True, seed=seed + epoch),
            desc='Training refine', leave=False,
        ):
            x_full = torch.from_numpy(batch['x']).to(dev)
            y = torch.from_numpy(batch['y']).to(dev)
            pm = torch.from_numpy(batch['padding_mask']).to(dev)
            lm = torch.from_numpy(batch['loss_mask']).to(dev)

            with torch.no_grad():
                coarse_logits = coarse(x_full, pm)
                coarse_soft = torch.softmax(coarse_logits, dim=-1)
            raw = x_full[:, :, :-4]
            prev = x_full[:, :, -4:]
            x_refine = torch.cat([raw, coarse_soft, prev], dim=-1).detach()

            optimizer.zero_grad(set_to_none=True)
            logits = refine(x_refine, pm)
            loss = _ce_smoothed(logits, y, lm, label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refine.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            if gstep % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                logger.info(f'  step {gstep}, loss={float(loss.item()):.4f}, lr={lr_now:.2e}')
            nb += 1
            gstep += 1
        train_loss = epoch_loss / max(nb, 1)

        refine.eval()
        total_correct, total_tokens = 0, 0
        with torch.inference_mode():
            for batch in build_batches(val_chunks, batch_size, shuffle=False):
                x_full = torch.from_numpy(batch['x']).to(dev)
                y = torch.from_numpy(batch['y']).to(dev)
                pm = torch.from_numpy(batch['padding_mask']).to(dev)
                lm = torch.from_numpy(batch['loss_mask']).to(dev)
                coarse_soft = torch.softmax(coarse(x_full, pm), dim=-1)
                raw = x_full[:, :, :-4]
                prev = x_full[:, :, -4:]
                x_refine = torch.cat([raw, coarse_soft, prev], dim=-1)
                logits = refine(x_refine, pm)
                preds = logits.argmax(dim=-1)
                mask = (lm & (y >= 0))
                total_correct += int(((preds == y.long()) & mask).sum().item())
                total_tokens += int(mask.sum().item())
        val_acc = total_correct / max(total_tokens, 1)
        logger.info(f'Epoch {epoch+1}: train_loss={train_loss:.4f} val_acc={val_acc*100:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            best_path = os.path.join(out_dir, f'{sd}-arrowlimbs_to_limb-torch-best.safetensors')
            save_safetensors(refine, best_path)
            with open(os.path.join(out_dir, f'{sd}-arrowlimbs_to_limb-torch-best.meta'), 'w') as f:
                json.dump({
                    'input_dim': raw_dim + REFINE_EXTRA_DIM + 4,  # raw + coarse_soft + prev_limb
                    'd_model': d_model, 'n_heads': n_heads,
                    'n_layers': n_layers, 'ffn_dim': ffn_dim,
                    'n_classes': 3, 'is_refine': True,
                    'coarse_weights': os.path.basename(coarse_path),
                }, f)
            logger.success(f'[torch] New best: {best_acc*100:.2f}%')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f'[torch] Early stop at epoch {epoch+1}')
                break

    logger.success(f'[torch] Refine training complete. Best={best_acc*100:.2f}%')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['mlx', 'torch'], default='mlx')
    p.add_argument('--singles_or_doubles', required=True, choices=['singles', 'doubles'])
    p.add_argument('--manual_chart_struct_folder', required=True,
                   help='Folder with .npz cache from cache_chunks.py')
    p.add_argument('--coarse_weights', required=True,
                   help='Path to coarse arrows_to_limb .safetensors')
    p.add_argument('--coarse_meta', required=True,
                   help='Path to coarse .meta JSON (sister of weights)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--limit_charts', type=int, default=None)
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--large_model', action='store_true')
    p.add_argument('--resume_from', default=None)
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default=None,
                   help='Torch only. MLX always uses unified memory.')
    p.add_argument('--coarse_init', action='store_true',
                   help='Torch only. Initialize refine weights from coarse checkpoint. '
                        'Copies matching tensors; input_proj gets raw+prev_limb from coarse, '
                        'coarse_softmax cols start at default init. Recommended: avoids '
                        'random-init degradation vs coarse baseline.')
    a = p.parse_args()

    kwargs = dict(
        folder=a.manual_chart_struct_folder, sd=a.singles_or_doubles,
        out_dir=a.out_dir, coarse_path=a.coarse_weights, coarse_meta_path=a.coarse_meta,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
        warmup_epochs=a.warmup_epochs, patience=a.patience,
        limit_charts=a.limit_charts, label_smoothing=a.label_smoothing,
        large_model=a.large_model, resume_from=a.resume_from,
    )
    if a.backend == 'mlx':
        _train_mlx(**kwargs)
    else:
        _train_torch(device=a.device, coarse_init=a.coarse_init, **kwargs)


if __name__ == '__main__':
    main()
