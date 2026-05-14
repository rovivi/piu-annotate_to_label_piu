#!/usr/bin/env bash
# Stages 2-4: singles coarse → singles refine → doubles refine
# Run AFTER doubles coarse (stage 1) finishes.
# Usage: bash scripts/run_stages_2_4.sh

set -e
export HSA_OVERRIDE_GFX_VERSION=10.3.0
mkdir -p logs artifacts/models

CACHE_S=artifacts/cache/torch-singles
CACHE_D=artifacts/cache/torch-doubles

echo "=== STAGE 2: singles coarse ==="
python cli/limbuse/train_torch.py \
  --singles_or_doubles singles \
  --manual_chart_struct_folder "$CACHE_S" \
  --out_dir artifacts/models/visss-torch-v9-singles \
  --epochs 30 --batch_size 8 --grad_accum_steps 4 --dtype bf16 \
  --large_model --lr 2e-4 --warmup_epochs 2 --patience 8 \
  --save_by_ar --device cuda \
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

echo "=== STAGE 5: match models (singles) ==="
python scripts/train_match_models.py \
  --sd singles \
  --cache_dir "$CACHE_S" \
  --out_dir artifacts/models/visss-torch-v9-singles \
  2>&1 | tee logs/train_match_singles.log

echo "=== STAGE 6: match models (doubles) ==="
python scripts/train_match_models.py \
  --sd doubles \
  --cache_dir "$CACHE_D" \
  --out_dir artifacts/models/visss-torch-v9-doubles \
  2>&1 | tee logs/train_match_doubles.log

echo "=== STAGE 7: generate benchmark baseline ==="
python scripts/gen_baseline.py \
  --csv_dir artifacts/manual-chartstructs/visss-120524 \
  --out artifacts/benchmark_baseline_local.json \
  --n 120

echo "=== STAGE 8: benchmark all models ==="
mkdir -p logs/comparator
python scripts/compare_models.py \
  --models "coarse_singles:artifacts/models/visss-torch-v9-singles" \
  --models "coarse_doubles:artifacts/models/visss-torch-v9-doubles" \
  --models "refine_singles:artifacts/models/visss-torch-v9-singles-refine" \
  --models "refine_doubles:artifacts/models/visss-torch-v9-doubles-refine" \
  --baseline artifacts/benchmark_baseline_local.json \
  --csv_dir artifacts/manual-chartstructs/visss-120524 \
  --output_json logs/comparator/benchmark_results.json \
  2>&1 | tee logs/benchmark.log

echo "=== ALL DONE ==="
echo "Singles coarse: $(grep 'New best' logs/train_singles_coarse.log | tail -1)"
echo "Doubles coarse: $(grep 'New best' logs/train_doubles_coarse.log | tail -1)"
echo "Singles refine: $(grep 'New best' logs/train_singles_refine.log | tail -1)"
echo "Doubles refine: $(grep 'New best' logs/train_doubles_refine.log | tail -1)"
echo "Match singles:  $(grep 'match_next\|match_prev' logs/train_match_singles.log | grep -E 'val_acc|AUC' | tail -2)"
echo "Match doubles:  $(grep 'match_next\|match_prev' logs/train_match_doubles.log | grep -E 'val_acc|AUC' | tail -2)"
echo ""
echo "--- Benchmark summary (from logs/benchmark.log) ---"
grep -E 'acc=|accuracy|Overall|pattern' logs/benchmark.log | tail -30
