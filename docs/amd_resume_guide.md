# AMD 6950 XT — Guía de Resume de Training

> **Audiencia:** Rodrigo, retomando training en la máquina con 6950 XT (Linux + ROCm).
> **Estado en el Mac:** v9-large-ft coarse (94.6% val) + v9-refine (92.75% val) entrenados.
>   Comparator dice v9_coarse = 93.43% en benchmark de 60 charts; refine no mejoró overall.
> **Hardware AMD real:** RX 6750 XT (RDNA2, Navi22), **12 GB VRAM**, RAM normal.
>   Linux, PyTorch ROCm 6.1 (`torch-2.6.0+rocm6.1`), Python 3.13.
>   El override `HSA_OVERRIDE_GFX_VERSION=10.3.0` funciona también para Navi22 (gfx1031).

---

## 0. Por qué tiene sentido seguir aquí

- M5 Pro hizo refine de singles en ~9 h. Demasiado lento.
- 6950 XT con ROCm acelera 3–5× a igual modelo, sobretodo en attention.
- VRAM 16 GB < unified-mem M5, pero podemos compensar con **bf16 + gradient accumulation**.
- Te conviene entrenar:
  1. Doubles coarse v9 (no existe todavía con feature schema actual).
  2. Singles coarse v10 — refinar el approach del refine que no funcionó (init from coarse).
  3. Refine variant con coarse-initialized refine.

---

## 1. Setup PyTorch ROCm (instalado ✅)

PyTorch ROCm ya instalado en base conda (Python 3.13):

```bash
# Si hay que reinstalar:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
pip install safetensors loguru tqdm numpy pandas lightgbm pyyaml scikit-learn
pip install -e .

# Verificar
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')
"
# → CUDA: True  Device: AMD Radeon RX 6750 XT  VRAM: 12.0 GB
```

> **Nota:** ROCm full stack NO necesario. El wheel de PyTorch bundlea HIP.
> Advertencia `hipBLASLt unsupported architecture` es esperada en Navi22 — cae a hipblas, funciona igual.

Otras deps:
```bash
pip install safetensors loguru tqdm numpy pandas lightgbm pyyaml scikit-learn
pip install -e .   # piu_annotate package
```

---

## 2. Estado de datos (LISTO ✅)

Todo disponible localmente:

| Qué | Path | Estado |
|---|---|---|
| Cache singles (24 dims) | `artifacts/cache/torch-singles/` | 2534 npz ✅ |
| Cache doubles (29 dims) | `artifacts/cache/torch-doubles/` | 1613 npz ✅ |
| CSVs ground-truth | `artifacts/manual-chartstructs/visss-120524/` | 4262 csv ✅ |

> Cache regenerado con featurizer actual (24/29 dims → +prev_limb = 28/33 modelo input).
> Cache antiguo (18/23 dims) era de featurizer viejo — no compatible con train_torch.py.

---

## 3. Plan de entrenamiento sugerido (orden de impacto)

### 3.1 Doubles coarse (PRIORIDAD 1) — no existe con schema v9

```bash
mkdir -p logs
HSA_OVERRIDE_GFX_VERSION=10.3.0 python cli/limbuse/train_torch.py \
  --singles_or_doubles doubles \
  --manual_chart_struct_folder artifacts/cache/mlx-v9-doubles \
  --out_dir artifacts/models/visss-torch-v9-doubles \
  --epochs 25 --batch_size 8 --grad_accum_steps 4 --dtype bf16 \
  --large_model --lr 2e-4 --warmup_epochs 2 --patience 6 \
  --save_by_ar \
  --device cuda \
  2>&1 | tee logs/train_doubles_coarse_amd.log &

# En otra terminal: dashboard en tiempo real
python scripts/train_dashboard.py --log logs/train_doubles_coarse_amd.log
```

`--save_by_ar`: guarda el checkpoint con mejor AR accuracy (autoregresiva, sin teacher forcing) → checkpoint más real para inferencia.

**Por qué estos hiperparámetros:**
- `batch_size 8 + grad_accum_steps 4` = batch efectivo 32, fits en 16 GB VRAM con bf16.
- `bf16` ahorra ~40% VRAM sin perdida de accuracy (vs fp16, bf16 evita scaler y es mejor en RDNA2).
- `large_model` (d_model=384, 8 layers, 14.7M params) — mejor calidad que el default 5M.
- 25 epochs con patience 6 → cortará temprano si no mejora.

**Tiempo esperado:** ~60–90 min en 6950 XT.

### 3.2 Singles coarse v10 — re-entrenar con torch para parity check

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python cli/limbuse/train_torch.py \
  --singles_or_doubles singles \
  --manual_chart_struct_folder artifacts/cache/mlx-v9-singles \
  --out_dir artifacts/models/visss-torch-v9-singles \
  --epochs 30 --batch_size 8 --grad_accum_steps 4 --dtype bf16 \
  --large_model --lr 2e-4 --warmup_epochs 2 --patience 8 \
  --save_by_ar \
  --device cuda \
  2>&1 | tee logs/train_singles_coarse_amd.log &

python scripts/train_dashboard.py --log logs/train_singles_coarse_amd.log
```

**Target:** matchear ~94.6% val_acc del v9-large-ft del Mac. Si llega ahí, el backend PyTorch+ROCm es viable end-to-end.

**Tiempo:** ~90–120 min.

### 3.3 Refine con coarse-init (FIX del problema de Mac)

El refine del Mac arrancó random → 92.75% (worse que coarse). Solución: **inicializar el refine con los pesos del coarse**, dejar que solo aprenda la diferencia. Bajo LR.

Actualmente `train_refine.py` no soporta init-from-coarse. Cambio necesario en `cli/limbuse/train_refine.py:_train_torch` después de `refine = build_model(...)`:

```python
# Initialize refine from coarse weights (shape mismatch only on input_proj
# because refine has +3 extra dims; copy what matches, init the new dims to 0).
from piu_annotate.ml.arch_torch import load_safetensors
coarse_state = {}
import safetensors.torch as st
coarse_state = st.load_file(coarse_path, device='cpu')

refine_state = refine.state_dict()
for k, v in coarse_state.items():
    if k not in refine_state:
        continue
    if v.shape == refine_state[k].shape:
        refine_state[k] = v
    elif k == 'input_proj.weight':
        # coarse input_proj: (d_model, raw+4)
        # refine input_proj: (d_model, raw+3+4) = coarse + 3 extra cols for coarse_softmax
        new_w = refine_state[k].clone()
        new_w[:, :v.shape[1]] = v[:, :input_dim_raw]                     # raw cols
        new_w[:, -4:] = v[:, -4:]                                         # prev_limb cols
        # middle (coarse_softmax) cols stay at default init
        refine_state[k] = new_w
refine.load_state_dict(refine_state)
logger.info('Refine initialized from coarse weights')
```

El patch ya está integrado como `--coarse_init`. Corre con LR muy bajo:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python cli/limbuse/train_refine.py \
  --backend torch --singles_or_doubles singles \
  --manual_chart_struct_folder artifacts/cache/mlx-v9-singles \
  --coarse_weights artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-torch-best.safetensors \
  --coarse_meta    artifacts/models/visss-torch-v9-singles/singles-arrows_to_limb-torch-best.meta \
  --out_dir artifacts/models/visss-torch-v9-singles-refine \
  --epochs 8 --batch_size 8 --lr 5e-5 --patience 3 \
  --coarse_init \
  --device cuda \
  2>&1 | tee logs/train_singles_refine_amd.log &

python scripts/train_dashboard.py --log logs/train_singles_refine_amd.log
```

**Target:** ≥ 95% val (mejorar sobre coarse 94.6%).

**Tiempo:** ~40–60 min.

### 3.4 Doubles refine (después de doubles coarse)

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python cli/limbuse/train_refine.py \
  --backend torch --singles_or_doubles doubles \
  --manual_chart_struct_folder artifacts/cache/mlx-v9-doubles \
  --coarse_weights artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-torch-best.safetensors \
  --coarse_meta    artifacts/models/visss-torch-v9-doubles/doubles-arrows_to_limb-torch-best.meta \
  --out_dir artifacts/models/visss-torch-v9-doubles-refine \
  --epochs 8 --batch_size 8 --lr 5e-5 --patience 3 \
  --coarse_init \
  --device cuda \
  2>&1 | tee logs/train_doubles_refine_amd.log &

python scripts/train_dashboard.py --log logs/train_doubles_refine_amd.log
```

---

## 4. Evaluación

Una vez tengas los `.safetensors`, evalúa contra el benchmark de 60 charts:

```bash
python scripts/compare_models.py \
  --models v9_mac:artifacts/models/visss-mlx-v9-large-ft \
  --models v9_torch:artifacts/models/visss-torch-v9-singles \
  --models v9_torch_refine:artifacts/models/visss-torch-v9-singles-refine \
  --baseline artifacts/benchmark_baseline_60.json \
  --csv_dir artifacts/manual-chartstructs/visss-120524-eaware \
  --output_json logs/compare_amd.json \
  --plot logs/compare_amd_confusion.png
```

Luego HTML:
```bash
python scripts/generate_comparison_report.py \
  --data logs/compare_amd.json \
  --out  logs/report_amd.html
```

Abre `report_amd.html` en navegador.

---

## 5. Si OOM

Síntoma típico: `RuntimeError: HIP out of memory`. Acciones por orden:

1. Bajar `--batch_size 8` → `4`, subir `--grad_accum_steps 4` → `8` (mismo efectivo, mitad de VRAM).
2. Cambiar `--dtype bf16` → `fp16` (más agresivo, ~50% memoria).
3. Si aún OOM: quitar `--large_model` (default 5M params vs 14.7M).
4. Reducir `MAX_SEQ_LEN` en `piu_annotate/ml/seq_data.py` de 1024 a 512 (recompactará chunks).

Para monitorear VRAM en tiempo real (otra terminal):
```bash
watch -n 1 'rocm-smi --showmemuse --showtemp'
```

---

## 6. Bugs conocidos / cosas a vigilar

- **`HSA_OVERRIDE_GFX_VERSION=10.3.0` siempre.** Sin él, PyTorch ROCm no detecta la 6950 XT como soportada (gfx1030 oficialmente parchada con override).
- **bf16 puede dar NaN si LR muy alto.** Si ves loss=NaN, baja LR a 1e-4 o cambia a fp32.
- **Pesos no son compatibles MLX ↔ Torch directamente.** Cada backend entrena desde cero. No hay converter aún.
- **Cache .npz transferible.** Sí. Solo numpy, agnostic.

---

## 7. Cuando termines

1. Sube los `.safetensors` + `.meta` a un dir accesible al Mac (o git LFS).
2. Reporta números del `compare_amd.json` al Mac para añadir al doc maestro.
3. Si singles_torch ≥ 94% en val, ya es viable correr inferencia en producción con `--model_backend torch`. Cambia `args['model.dir']` apuntando al dir torch.

---

## 8. Comandos copy-paste resumen

```bash
# 1) Setup
export HSA_OVERRIDE_GFX_VERSION=10.3.0
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install safetensors loguru tqdm

# 2) Doubles coarse (priority 1)
python cli/limbuse/train_torch.py --singles_or_doubles doubles \
  --manual_chart_struct_folder artifacts/cache/mlx-v9-doubles \
  --out_dir artifacts/models/visss-torch-v9-doubles \
  --epochs 25 --batch_size 8 --grad_accum_steps 4 --dtype bf16 \
  --large_model --lr 2e-4 --device cuda \
  2>&1 | tee logs/doubles_coarse.log

# 3) Singles coarse parity
python cli/limbuse/train_torch.py --singles_or_doubles singles \
  --manual_chart_struct_folder artifacts/cache/mlx-v9-singles \
  --out_dir artifacts/models/visss-torch-v9-singles \
  --epochs 30 --batch_size 8 --grad_accum_steps 4 --dtype bf16 \
  --large_model --lr 2e-4 --device cuda \
  2>&1 | tee logs/singles_coarse.log

# 4) Compare
python scripts/compare_models.py \
  --models v9_torch:artifacts/models/visss-torch-v9-singles \
  --baseline artifacts/benchmark_baseline_60.json \
  --csv_dir artifacts/manual-chartstructs/visss-120524-eaware \
  --output_json logs/compare_amd.json --plot logs/compare_amd.png

# 5) Report
python scripts/generate_comparison_report.py --data logs/compare_amd.json --out logs/report_amd.html
```

Listo. Cualquier duda revisa también `docs/next_gen_architecture.md` para el contexto completo del plan.
