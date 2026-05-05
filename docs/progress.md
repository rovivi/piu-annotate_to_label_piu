# PIU Limb Annotation — Progress Guide

## Estado actual (2026-05-03)

### Accuracy global (benchmark vs vis-ss, 3702 charts)

| Métrica | Modelo viejo | Modelo nuevo | Delta |
|---|---|---|---|
| Tap overall | 82.3% | **96.4%** | +14.1pp |
| Jacks/repeated | 71.2% | **93.4%** | +22.2pp |
| Triple taps | 83.0% | **86.9%** | +3.9pp |
| Hold accuracy | 73.5% | **97.0%** | +23.5pp |

### Archivos generados hoy

| Archivo | Descripción |
|---|---|
| `artifacts/models/visss/singles-*.txt` | 4 modelos LightGBM singles re-entrenados |
| `artifacts/models/visss/doubles-*.txt` | 4 modelos LightGBM doubles re-entrenados |
| `artifacts/models/visss/training_curves_singles.png` | Curvas de entrenamiento singles |
| `artifacts/models/visss/training_curves_doubles.png` | Curvas de entrenamiento doubles |
| `artifacts/processed_db/` | 4372 charts re-inferidos con nuevos modelos |
| `artifacts/benchmark_charts_newmodel.png` | Charts del benchmark (4 paneles) |
| `artifacts/model_comparison.png` | Comparación old vs new en 54 charts baseline |
| `artifacts/benchmark_baseline_60.json` | Baseline: 30 peores accuracy + 30 más difíciles |

---

## MLX Transformer Migration (2026-05-04) — SUCCESSFUL

**Status:** Training completed successfully.
- Singles: ~15 mins, Max RAM ~420 MB, Validation Acc: 79.7%
- Doubles: ~16 mins, Max RAM ~550 MB, Validation Acc: 70.3%
Checkpoints saved and loadable using Safetensors.

### Optimizations Applied
- Pre-cached feature extraction chunks to disk (`.npz`)
- Reordered batches by length (bucketing)
- Compiled the backward and forward pass natively with `@mx.compile(inputs=state, outputs=state)`
- Properly broadcasted attention masks and split validation correctly.

### What Was Built

| Component | File | Status |
|---|---|---|
| Mirror augmentation | `piu_annotate/formats/mirror.py` | ✅ Working |
| Transformer (4L, 8H, d=128, 3-class) | `piu_annotate/ml/mlx_architecture.py` | ✅ Built, not tested |
| `LimbLabel.from_limb_annot` (e→2) | `piu_annotate/ml/datapoints.py` | ✅ Fixed from e→0 |
| Cross-entropy loss (manual log-softmax) | `train_mlx.py` | ✅ Working |
| Flat params + np.savez | `train_mlx.py` | ⚠️ **BUG: saves `.npz` as `.safetensors`** |

### Critical Bugs Fixed

1. **BUG 1 (HIGH):** Replaced custom `np.savez` logic with `mx.save_safetensors` to ensure model files are readable.
2. **BUG 2 (HIGH):** Fixed mirror augmentation to correctly slice dimensions without feature corruption.
3. **BUG 3 (MEDIUM):** Fixed attention mask shape to explicitly broadcast over queries vs keys correctly using `[:, None, None, :]`.
4. **BUG 4 (MEDIUM):** Eliminated OOM by generating lazy caches to disk (`cache_chunks.py`) and iterating batches directly from `.npz` files.

---

## Charts con problemas conocidos

### Canciones no en master_db (no aparecen en piulatam)
- **Butterfly** — existe en vis-ss pero no en ligas master_db. Comportamiento esperado.

### Canciones con 0 charts procesados (no hay match en vis-ss)
- `Ugly_Dee` (D17, D18, S3) — benchmark muestra n/a, no hay ground truth

### Songs que fallaban (bugs arreglados hoy)
- `Mopemope D24`, `Naissance S5`, `Vook S18` — `ssc_to_chartstruct` retornaba 2 valores en vez de 3 en path de error → **arreglado** en `piu_annotate/formats/ssc_to_chartstruct.py:98`
- `Top City S20/D21` — numpy int64 no serializable en JSON → **arreglado** en `piu_annotate/formats/chart.py:135` y `cli/ingest/process_db_matches.py`
- `Mopemope D27` — chart corrupto en .ssc (error vacío), skip esperado

### Canciones que no hacen match por nombre
- `CROSS RAY (feat. 月下Lia)` — caracteres japoneses rompen fuzzy match
- `"Simon Says` — comilla en el nombre rompe fuzzy match

---

## Charts especialmente malos con el modelo nuevo

Los peores (tap < 85%) son todos S18-S24 con jacks complejos:

| Chart | Tap | Jack | Notas |
|---|---|---|---|
| `Destination_SHK_D20_SHORTCUT` | 53.2% | 62.5% | Jack patterns en doubles |
| `The_End_of_the_World_MonstDeath_S20` | 60.5% | 74.3% | Patrón S20 inusual |
| `Loki_Lotze_S21` | 66.0% | 55.6% | Triple + jack combinado |
| `Imagination_SHK_S18` | 68.5% | 81.1% | Estructura SHK rara |
| `8_6_DASU_D21` | 73.3% | 61.7% | Jacks doubles lv21 |
| `Ultimatum_D27` | 87.6% | **58.7%** | Jack en D27 — punto débil principal |

**Patrón**: los peores son charts con jacks largos en S18-S24 y charts con estructura rítmica SHK/DASU.

---

## Próximos pasos planeados

### Fase 1 — Hard Negative Mining + context_len=32
- Correr inferencia sobre todos los CSVs de entrenamiento
- Identificar los 100 charts con más errores
- Upsample 3x en el dataset de training
- Aumentar `ft.context_length` de 20 → 32 en `piu_annotate/ml/featurizers.py`
- Re-entrenar y comparar vs benchmark actual

**Estimado**: +3-5pp en jacks, +1-2pp overall

### Fase 2 — Transformer correcto (siguiente sesión)
Reemplazar LightGBM con transformer de secuencia real:
- Input: secuencia de 32 arrows consecutivos, cada uno con ~35 features raw
- Arquitectura: 3 capas, 4 heads, model_dim=128, mlp_dims=512
- Training stride: 8 (eficiencia + augmentation implícita)
- Inference stride: 1 (calidad máxima)
- NO chunks de 512 — 32 es suficiente para patrones de PIU

**Estimado**: +3-5pp adicionales sobre Fase 1, especialmente en jacks largos

### Fase 3 — Opcional
- Ensemble LightGBM + Transformer (promedio de probabilidades) → +1-2pp

---

## Pipeline completo

```
vis-ss chart-jsons/120524/
    ↓ scripts/visss_to_chartstruct.py
artifacts/manual-chartstructs/visss-120524/  (4261 CSVs con Limb annotation)
    ↓ cli/limbuse/train_lgbm.py
artifacts/models/visss/  (8 modelos: singles/doubles × 4 tasks)
    ↓ cli/ingest/process_db_matches.py
artifacts/processed_db/  (4372 JSONs)
    ↓ sync_to_piulatam.py
piulatam/public/chart-jsons/  (visor web)

    ↓ cli/limbuse/infer_v8_batch.py
comparations/  (index.json + generated/ + origin_viss/)
    ↓ python3 -m http.server 8080
comparations/comparation.html  (comparison viewer side-by-side)
```

## Comandos útiles

```bash
# Re-entrenar singles
python3 cli/limbuse/train_lgbm.py --singles_or_doubles singles

# Re-entrenar doubles  
python3 cli/limbuse/train_lgbm.py --singles_or_doubles doubles

# Regenerar processed_db
python3 cli/ingest/process_db_matches.py

# Benchmark completo con charts
python3 scripts/benchmark_annotations.py --show_worst 30 --top_diff 30 --plot

# Comparar old vs new en 54 charts baseline
python3 scripts/compare_models.py --plot

# Sync a piulatam
python3 sync_to_piulatam.py

# Generar datos para el comparison viewer (v8 vs vis-ss)
python3 cli/limbuse/infer_v8_batch.py \
  --simfiles_dir /Users/rodrigo/dev/piu/piu_sim_files \
  --out_dir comparations/generated \
  --model_dir artifacts/models/visss-mlx-v8 \
  --viss_src /Users/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524 \
  --sd both

# Servir comparison viewer
# Requisito: ln -s piu-vis-ss_for_piumx/public/images comparations/images
cd /Users/rodrigo/dev/piu/piu-annotate_to_label_piu/comparations && python3 -m http.server 8080
```
