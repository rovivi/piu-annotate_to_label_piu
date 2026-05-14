# Torch → MLX Inference Guide

Models trained on AMD (ROCm/PyTorch) can be converted to MLX for fast inference on Apple Silicon.

## Why bother

MLX on M5 Pro benchmarks all 4000+ charts in <30 min.
PyTorch/ROCm on RX 6750 XT takes ~70 h for the same with full beam search.

---

## Step 1: Pull latest weights from this repo

```bash
git pull origin main
```

The trained safetensors are NOT committed (too large). Copy them from the AMD machine:

```bash
# From AMD machine — rsync models to Mac
rsync -avz --progress \
  artifacts/models/visss-torch-v9-singles/ \
  <mac>:~/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss-torch-v9-singles/

rsync -avz --progress \
  artifacts/models/visss-torch-v9-doubles/ \
  <mac>:~/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss-torch-v9-doubles/
```

---

## Step 2: Convert weights (run on Mac)

```bash
cd ~/dev/piu/piu-annotate_to_label_piu

python scripts/convert_torch_to_mlx.py \
  --torch_weights artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-torch-best.safetensors \
  --out           artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-mlx-best.safetensors \
  --sd singles --verify

python scripts/convert_torch_to_mlx.py \
  --torch_weights artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-torch-best.safetensors \
  --out           artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-mlx-best.safetensors \
  --sd doubles --verify
```

`--verify` runs a smoke test forward pass through the MLX model. If it prints `MLX forward pass OK`, you're good.

**If verify fails with a key mismatch**, check the actual MLX weight names:

```python
import mlx.nn as nn
m = nn.TransformerEncoder(num_layers=1, dims=384, num_heads=8, mlp_dims=1536)
flat, _ = m.flatten_module()
for k in list(flat.keys())[:15]:
    print(k)
```

Then update the key mapping in `scripts/convert_torch_to_mlx.py` accordingly.

---

## Step 3: Run full benchmark on all charts (Mac)

```bash
python scripts/compare_models.py \
  --models "v9_singles:artifacts/models/visss-torch-v9-singles" \
  --models "v9_doubles:artifacts/models/visss-torch-v9-doubles" \
  --baseline artifacts/benchmark_baseline_local.json \
  --csv_dir artifacts/manual-chartstructs/visss-120524 \
  --output_json logs/comparator/benchmark_full_mlx.json
```

For ALL ~4262 charts (no baseline sampling):

```bash
# Generate full baseline from all CSVs
python scripts/gen_baseline.py \
  --csv_dir artifacts/manual-chartstructs/visss-120524 \
  --out artifacts/benchmark_all_charts.json \
  --n 4262

python scripts/compare_models.py \
  --models "v9_singles:artifacts/models/visss-torch-v9-singles" \
  --models "v9_doubles:artifacts/models/visss-torch-v9-doubles" \
  --baseline artifacts/benchmark_all_charts.json \
  --csv_dir artifacts/manual-chartstructs/visss-120524 \
  --output_json logs/comparator/benchmark_all.json
```

---

## What was trained (AMD results)

| Stage | Best AR val_acc |
|-------|----------------|
| Singles coarse | 94.5% |
| Doubles coarse | 93.3% |
| Singles refine | 94.94% |
| Doubles refine | 93.78% |
| Match next/prev singles | 98.1% AUC 0.996 |
| Match next/prev doubles | 98.2% AUC 0.995 |

### End-to-end benchmark (30 charts, full pipeline)

| Pattern | Singles | Doubles |
|---------|---------|---------|
| **all** | **95.3%** | **95.7%** |
| tap | 96.1% | 97.0% |
| stream | 96.7% | 97.6% |
| jack | 90.7% | 92.4% |
| triple | 77.8% | 92.7% |
| bracket_rr | 63.5% | 92.7% |
| jump | 96.8% | 86.8% |

Weak spots: singles bracket_rr (63%) and triple (78%).

---

## Model architecture

Same `LimbSequenceTransformerTorch` / `LimbSequenceTransformer` (MLX) architecture:
- d_model=384, n_heads=8, n_layers=8, ffn_dim=1536 → 14.28M params
- Two-pass: coarse (`arrows_to_limb`) → refine (`arrowlimbs_to_limb`)
- Match models: LightGBM binary classifiers for match_next / match_prev (pairwise features)
- Cache dims: singles=28 (24 raw + 4 prev_limb), doubles=33 (29 raw + 4 prev_limb)
