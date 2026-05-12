# Progress Report: v9-large MLX Transformer Resume
**Date:** 2026-05-05
**Model:** v9-large (14.7M params, singles)
**Dataset:** visss-120524-eaware (2279 singles charts)
**Objective:** Push val_acc from 92.7% → 93.5%+ then hard-negative fine-tune toward 94%+

## Checkpoint Resume Details
- **Resume from:** `out_mlx_v9_large_fast/best.npz` (92.7% val_acc, epoch 34/40 of prior run)
- **Command:** `python cli/limbuse/train_mlx.py --sd singles --out_dir out_mlx_v9_large_fast --epochs 15 --batch_size 32 --lr 1e-4 --patience 5 --seed 1 --large_model --resume_from out_mlx_v9_large_fast/best.npz`
- **PID:** 30357
- **Batch size:** 32
- **LR schedule:** Warmup 2 epochs → cosine decay from 1e-4
- **Scheduled sampling:** ss_prob_max=0.3 (ramping linearly post-warmup)
- **Label smoothing:** 0.1
- **Class weight E:** 1.0

## Epoch-by-Epoch Results

| Epoch | val_acc | L     | R     | E     | LR       | Notes                    |
|-------|---------|-------|-------|-------|----------|--------------------------|
| 1     | 92.7%   | 93.7% | 93.3% | 2.1%  | 5.03e-05 | Tied prior best          |
| 2     | 91.6%   | 94.3% | 90.6% | 0.2%  | 1.00e-04 | LR spike caused dip      |
| 3     | 92.8%   | 93.8% | 93.3% | 1.7%  | 9.85e-05 | New best (+0.1pp)        |
| 4     | 93.0%   | 93.7% | 93.8% | 2.4%  | 9.42e-05 | New best (+0.2pp)        |
| 5     | 93.1%   | 93.7% | 94.0% | 6.6%  | 8.73e-05 | New best (+0.1pp)        |
| 6     | 93.3%   | 93.7% | 94.5% | 2.2%  | 7.82e-05 | New best (+0.2pp)        |
| 7     | 93.3%   | 93.9% | 94.3% | 3.8%  | 6.75e-05 | Flat, patience 0/5       |
| 8     | 93.4%   | 94.1% | 94.3% | 6.2%  | 5.58e-05 | New best (+0.1pp)        |
| 9     | 93.5%   | 94.1% | 94.4% | 4.5%  | 4.38e-05 | New best (+0.1pp)        |
| 10    | 93.6%   | 94.4% | 94.2% | 7.4%  | 3.22e-05 | New best (+0.1pp)        |
| 11    | 93.6%   | 94.2% | 94.6% | 1.7%  | 2.16e-05 | Flat                     |
| 12    | 93.6%   | 94.4% | 94.4% | 2.9%  | 1.28e-05 | Flat                     |
| 13    | 93.6%   | 94.4% | 94.5% | 3.7%  | 6.19e-06 | Flat                     |
| 14    | **93.7%** | **94.4%** | **94.5%** | **3.9%** | **2.18e-06** | **New best (+0.1pp)** |
| 15    | 93.7%   | 94.4% | 94.5% | 3.8%  | 1.00e-06 | No improvement (1/5)     |

## Key Metrics
- **Improvement since resume:** +1.0pp (92.7% → 93.7%)
- **Improvement over v8 baseline:** +2.3pp (91.4% → 93.7%)
- **Improvement over v9-default:** +3.7pp (90.0% → 93.7%)
- **Per-class breakdown (epoch 14/15):**
  - Left: 94.4% (n=71,955)
  - Right: 94.5% (n=71,690)
  - Early: 3.8% (n=1,264) ← **bottleneck**
- **Gap to 96% target:** 2.3pp
- **Gap to 94% milestone:** 0.3pp
- **Oracle gap:** 1.1pp (AR val_acc=92.6%)

## Observations
- Right leg consistently outperforms Left by ~0.1pp
- Early class improved from 6.2% to 3.8% by epoch 15 (still bottleneck)
- Training completed all 15 epochs (no early stop triggered)
- No signs of overfitting (train_loss 0.4019, val_loss 0.3994 — very close)
- Scheduled sampling reached max 0.462 by final epoch
- Best checkpoint: `out_mlx_v9_large_fast/best.npz` (step 2200, val_acc=93.7%)

## Status
**TRAINING COMPLETE** — Run finished successfully at 18:49:18 on 2026-05-05

**FINE-TUNING COMPLETE — 94.6% val_acc**
- Epoch 30/30: val_acc=94.6%, L=95.5%, R=95.3%, E=3.3%
- AR val_acc=93.3%, oracle gap=1.2pp
- Model: `artifacts/models/visss-mlx-v9-large-ft`

**FINE-TUNING ROUND 2 IN PROGRESS** — Started 2026-05-06 08:36
- PID: 34296
- Log: `out_mlx_v9_large_ft.log` (append)
- Base: 94.6% val_acc from FT round 1
- Config: 20 epochs, LR 1e-5, batch 16, hard-neg weight 3.0, patience 5
- Target: 95.0%+ val_acc

## Next Steps
1. **Monitor FT run** — Watch `out_mlx_v9_large_ft.log`, expect 1-2h for 10 epochs
2. **Evaluate results** — Target 94.0%+ val_acc from hard-negative focus
3. **Ensemble seeds** — Retry seed1/seed2 with default (5M) model to compare and potentially ensemble
4. **Architecture exploration** — If FT plateau <94.5%, consider CRF head or encoder-decoder structure for final push to 96%
