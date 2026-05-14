#!/usr/bin/env bash
# Full training sequence: doubles coarse → singles coarse → singles refine → doubles refine
# Runs each stage in series; each uses GPU fully.
# Usage: bash scripts/run_all_training.sh
# Dashboard in another terminal: python scripts/train_dashboard.py --log logs/<current>.log

set -e
export HSA_OVERRIDE_GFX_VERSION=10.3.0
mkdir -p logs artifacts/models

CACHE_D=artifacts/cache/torch-doubles
CACHE_S=artifacts/cache/torch-singles

COMMON="--epochs 30 --batch_size 8 --grad_accum_steps 4 --dtype bf16 --large_model --lr 2e-4 --warmup_epochs 2 --patience 8 --save_by_ar --device cuda"

echo "=== STAGE 1: doubles coarse ==="
python cli/limbuse/train_torch.py \
  --singles_or_doubles doubles \
  --manual_chart_struct_folder "$CACHE_D" \
  --out_dir artifacts/models/visss-torch-v9-doubles \
  $COMMON \
  2>&1 | tee logs/train_doubles_coarse.log

echo "=== STAGE 2: singles coarse ==="
python cli/limbuse/train_torch.py \
  --singles_or_doubles singles \
  --manual_chart_struct_folder "$CACHE_S" \
  --out_dir artifacts/models/visss-torch-v9-singles \
  $COMMON \
  2>&1 | tee logs/train_singles_coarse.log

echo "=== STAGE 3: singles refine (coarse-init) ==="
python cli/limbuse/train_refine.py \
  --backend torch \
  --singles_or_doubles singles \
  --manual_chart_struct_folder "$CACHE_S" \
  --coarse_weights artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-torch-best.safetensors \
  --coarse_meta    artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-torch-best.meta \
  --out_dir artifacts/models/visss-torch-v9-singles-refine \
  --epochs 12 --batch_size 8 --lr 5e-5 --patience 4 \
  --large_model --coarse_init --device cuda \
  2>&1 | tee logs/train_singles_refine.log

echo "=== STAGE 4: doubles refine (coarse-init) ==="
python cli/limbuse/train_refine.py \
  --backend torch \
  --singles_or_doubles doubles \
  --manual_chart_struct_folder "$CACHE_D" \
  --coarse_weights artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-torch-best.safetensors \
  --coarse_meta    artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-torch-best.meta \
  --out_dir artifacts/models/visss-torch-v9-doubles-refine \
  --epochs 12 --batch_size 8 --lr 5e-5 --patience 4 \
  --large_model --coarse_init --device cuda \
  2>&1 | tee logs/train_doubles_refine.log

echo "=== ALL STAGES DONE ==="
echo "Singles coarse best:  $(grep 'New best' logs/train_singles_coarse.log | tail -1)"
echo "Doubles coarse best:  $(grep 'New best' logs/train_doubles_coarse.log | tail -1)"
echo "Singles refine best:  $(grep 'New best' logs/train_singles_refine.log | tail -1)"
echo "Doubles refine best:  $(grep 'New best' logs/train_doubles_refine.log | tail -1)"
