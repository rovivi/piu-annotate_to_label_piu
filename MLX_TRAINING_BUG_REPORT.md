# Bug Investigation Report: MLX Training Pipeline Crash

**Date:** 2026-05-04
**PID:** 92111
**Status:** Silent crash — no output files, no error logs, process disappeared
**Training Target:** LimbSequenceTransformer (3-class: left/right/either) on PIU singles charts

---

## 1. Executive Summary

PID 92111 ran for ~18 minutes and disappeared without producing output files or error logs. There is **no crash trace**. The training script has no file-based logging — all output was `logger.info()` to stdout, which was never captured or redirected.

**Root cause: Unknown.** The crash was silent. This report identifies the most likely candidates and documents the pipeline's latent bugs for future investigation.

---

## 2. What We Know

| Fact | Detail |
|------|--------|
| Process PID | 92111 |
| Working directory | `/Users/rodrigo/dev/piu/piu-annotate_to_label_piu` |
| Data location | `artifacts/manual-chartstructs/visss-120524-eaware/` (4261 CSV files) |
| Model | `LimbSequenceTransformer` (4 layers, 8 heads, d_model=128, 3 classes) |
| Training args | `--epochs 20 --batch_size 16 --lr 3e-4 --seed 0` |
| Output dir | `artifacts/models/visss-mlx/` (empty after crash) |
| Duration | ~18 minutes |
| Exit | No exit code captured, no core dump, no error log |

---

## 3. Pipeline Architecture

```
CSV Files (4261 charts)
    │
    ▼
ChartStruct.from_file()           [chart.py]
    │
    ▼
ChartStructFeaturizer(cs)         [featurizers.py]
    │  ArrowDataPoint → 18 dims (singles) or 21 dims (doubles)
    ▼
chartstruct_to_sequence()         [train_mlx.py:101-107]
    │  Returns: x (N, 18/21), y (N,)
    ▼
make_chunks()                     [train_mlx.py:110-119]
    │  Strided: max_len=1024, stride=896 (overlap=128)
    ▼
load_all_chunks()                 [train_mlx.py:122-152]
    │  Optional mirror augmentation (50% prob)
    ▼
build_batches()                   [train_mlx.py:155-186]
    │  Pad to max_len, build padding_mask, loss_mask
    ▼
LimbSequenceTransformer          [mlx_architecture.py]
    │  input_proj → sinusoidal PE → TransformerEncoder → out_head(3)
    ▼
compute_loss() / compute_accuracy()
    │
    ▼
optimizer.update() + mx.eval()
    │
    ▼
np.savez(out_dir/{sd}-arrows_to_limb-mlx-best.safetensors, **flat_params)
```

---

## 4. Critical Bugs Found

### BUG 1: Wrong File Format — Saves `.npz` as `.safetensors`

**Severity:** HIGH — Model not loadable after save
**Location:** `train_mlx.py:299-310`

```python
save_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
state = model.parameters()
flat_state = _flatten_params(state)
np.savez(save_path, **flat_state)  # WRONG: saves .npz, not .safetensors
```

**Problem:** The file extension says `.safetensors` but `np.savez` produces a `.npz` file. Loading with `mx.load(save_path)` would fail.

**Fix:**
```python
from mlx.core import save
save(save_path, model.parameters())
# OR
from safetensors.numpy import save_file
save_file(flat_state, save_path)
```

---

### BUG 2: Mirror Augmentation Feature/Label Mismatch

**Severity:** HIGH — Corrupts training data silently
**Location:** `train_mlx.py:138-144`

```python
if mirror_prob > 0 and rng.random() < mirror_prob:
    mirrored_cs = mirror_chartstruct(cs)
    mx2, my = chartstruct_to_sequence(mirrored_cs)
    min_len = min(len(cx), len(mx2))
    cx[:min_len] = mx2[:min_len]   # Overwrites original features
    cy[:min_len] = my[:min_len]    # Uses mirrored labels
    was_mirrored = True
```

**Problem:** After mirroring, `cx` now contains **mirrored features** but `cy` contains **mirrored labels**. The original chunk's features were overwritten. If mirrored chart has different arrow count, `min_len` truncates. This creates **feature-label misalignment** — the model learns incorrect associations.

**Fix:** Either use mirrored features + mirrored labels consistently, OR original features + original labels. The current code mixes them.

---

### BUG 3: Attention Mask Shape Mismatch

**Severity:** MEDIUM — May cause incorrect attention behavior
**Location:** `train_mlx.py:89-92` and `mlx_architecture.py:52-55`

```python
if mx.sum(padding_mask) > 0:
    attn_mask = mx.where(padding_mask[:, None, :, None], -1e9, 0.0)
```

**Problem:** `padding_mask` is `(B, max_len)`. The indexing produces a 4D tensor `(B, 1, L, 1)` which does not broadcast correctly to the transformer's expected attention mask shape `(B, H, L, L)`. The conditional `mx.sum(padding_mask) > 0` is also almost always true in variable-length batches, so attention masking is always applied.

**Fix:**
```python
attn_mask = mx.where(padding_mask[:, None, :], -1e9, 0.0)  # (B, 1, L)
```

---

### BUG 4: All Chunks Materialized in Memory Before Training

**Severity:** MEDIUM — Risk of OOM on large datasets
**Location:** `train_mlx.py:127`

```python
all_chunks = []
for cs, label_col in zip(loaded_charts, itertools.cycle([limb_col])):
    # ...
    all_chunks.append((cx, cy))
```

**Problem:** All chunks from all 4261 charts are loaded into a Python list before any training begins. With ~4261 charts producing 1-10+ chunks each, this can consume many gigabytes of RAM before training even starts.

**Fix:** Use a generator-based data loader that yields batches lazily, or preprocess and save chunks to disk.

---

### BUG 5: Silent Chart Skipping During Loading

**Severity:** LOW — Data loss (~10% of charts)
**Location:** `train_mlx.py:150`, `featurizers.py:127-131`

```python
except (ValueError, IndexError) as e:
    logger.warning(f"Skipping chart: {path} — {e}")
```

**Problem:** Charts with malformed `Limb annotation` strings (e.g., length mismatch with arrow count) are skipped silently. This produces 500+ warnings during loading. While handled gracefully, the user is not informed of the total count or which charts.

---

## 5. MLX-Specific Observations

| Pattern | Location | Status |
|---------|----------|--------|
| `mx.eval(model.parameters(), optimizer.state)` | Line 244 | OK — forces evaluation |
| `nn.value_and_grad(model, ...)` | Line 236 | OK |
| `mx.random.seed(seed)` | Line 268 | OK |
| Manual log-softmax (no `mx.log_softmax`) | Line 194 | OK — numerically stable shift |
| `nn.TransformerEncoder` with `checkpoint=True` | mlx_architecture.py | OK — memory efficient |

---

## 6. Why Did It Crash Silently?

### Hypothesis 1: Memory Exhaustion (Most Likely)

The process reached 5.5GB virtual memory before disappearing. On M-series Macs with unified memory, hitting the memory limit causes SIGKILL (exit 9) with no signal handler, hence no core dump or error output.

**Supporting evidence:**
- VSIZE grew from 1.3GB → 5.5GB over 18 minutes
- No output files written (OOM likely occurred during first save or late in epoch 1)
- macOS can kill processes silently when memory pressure

### Hypothesis 2: Save Operation Failure

If OOM occurred during the first `np.savez()` call (writing ~300MB of params), the write could fail silently.

### Hypothesis 3: Python Exception in C Extension

MLX operations run in C++/MPS. A NaN/Inf gradient could cause a segfault in the underlying MLX library, which would propagate as a silent process death.

---

## 7. Recommended Investigation Steps

### Immediate

1. **Re-run with memory monitoring:**
   ```bash
   /usr/bin/time -l python cli/limbuse/train_mlx.py ... 2>&1 | tee training.log
   ```
   Look for `maximum resident set size` to confirm memory usage.

2. **Run with ulimit:**
   ```bash
   ulimit -v 8388608  # ~8GB virtual
   python cli/limbuse/train_mlx.py ...
   ```

3. **Add `print()` at epoch boundaries** — if the crash happens during first save, we'll see epoch 1 complete, then nothing.

### Short-Term Fixes

1. **Fix safetensor save** (BUG 1)
2. **Fix mirror augmentation** (BUG 2)
3. **Add batch-level logging** to confirm it's not an epoch 0 crash
4. **Add `gc.collect()` after each epoch** to reduce memory pressure
5. **Save a minimal test** before full training to confirm model can forward-pass

### Long-Term Improvements

1. **Generator-based data loading** — don't materialize all chunks
2. **Preprocess and cache** chunks to `.npz` files
3. **Memory profiling** with `memory_profiler` or `tracemalloc`
4. **Graceful SIGKILL handling** — write checkpoint before potential OOM

---

## 8. Previous Successful Run (Reference)

A previous run completed all 20 epochs and produced:

```
2026-05-04 13:05:16 - 13:09:46  [Loading train chunks - 38 min]
2026-05-04 13:09:57 - 13:43:07  [Training - 33 min]
Epoch 20/20: train_loss=0.4249, val_loss=0.4293, val_acc=80.4%
Training complete. Best acc: 80.5%
```

**This run produced no output files either**, suggesting BUG 1 (wrong save format) prevented successful saves in all runs.

---

## 9. Summary Table

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| `.npz` as `.safetensors` | HIGH | train_mlx.py:302 | Model not loadable |
| Mirror augmentation x/y mismatch | HIGH | train_mlx.py:138-144 | Corrupted training data |
| Attention mask shape | MEDIUM | train_mlx.py:89-92 | Suboptimal training |
| All chunks in memory | MEDIUM | train_mlx.py:127 | OOM on large datasets |
| 537 skipped charts (~10%) | LOW | train_mlx.py:150 | Data loss |

---

## 10. Files Involved

| File | Purpose |
|------|---------|
| `cli/limbuse/train_mlx.py` | Training script (341 lines) |
| `piu_annotate/ml/mlx_architecture.py` | `LimbSequenceTransformer` |
| `piu_annotate/ml/datapoints.py` | `LimbLabel.from_limb_annot` |
| `piu_annotate/ml/models.py` | `MLXModel` wrapper |
| `piu_annotate/formats/mirror.py` | Mirror augmentation |
| `piu_annotate/format/chart.py` | `ChartStruct.from_file` |
| `piu_annotate/ml/featurizers.py` | `ChartStructFeaturizer` |
| `artifacts/manual-chartstructs/visss-120524-eaware/` | 4261 CSV charts |
| `artifacts/models/visss-mlx/` | Output directory (empty) |