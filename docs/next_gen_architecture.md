# Next-Gen Architecture — Doble Backend, Predicción en Dos Pasos, Comparador Pro

> **Documento de diseño. NO ejecutar entrenamientos desde este doc.**
> **Audiencia:** agente que va a implementar esto sin contexto previo + tú (Rodrigo).
> **Hardware objetivo:** Apple M5 Pro (MLX) + AMD 6950 XT (ROCm/PyTorch) + cualquier CPU.
> **Estado actual:** MLX-only, 91.4% singles (v8), 96.5% doubles. Plan eleva el techo y abre AMD.

---

## 0. Resumen ejecutivo (1 minuto)

Tres bloques de trabajo. Independientes pero compatibles.

| Bloque | Para qué | Cambio neto |
|---|---|---|
| **A. Backend dual MLX + PyTorch** | Entrenar en 6950 XT (ROCm) + Mac (MLX) + CUDA bonus | +1 archivo `arch_torch.py`, +1 `train_torch.py`, `models.py` con switch |
| **B. Two-pass limb (coarse → refine)** | Salto > v8: refinamiento condicionado, no scheduled sampling | +1 cabeza, +1 stage de training (fine-tune); modelo principal sin cambios |
| **C. Comparador 2.0** | Per-pattern + confusion + seek | Reescribir `comparation.html` partes + `compare_models.py` métricas |

Backbone compartido: featurizer (`featurizers.py`), datapoints (`datapoints.py`), cache .npz (`cache_chunks.py`), tactician post-proc (`tactics.py`). **Esto no se toca.** El motor de entrenamiento es lo único que se duplica.

Decisión clave: el **modelo es contrato (input_dim, n_clases, layer-sizes, weight names)**, no implementación. Si el contrato es idéntico, los `.safetensors` MLX cargan en PyTorch y viceversa con un script de conversión de 20 LOC.

---

## 1. Diagnóstico del estado actual

### 1.1 Stack ML actual

```
data: artifacts/manual-chartstructs/visss-120524-eaware/*.csv
  ↓ ChartStructFeaturizer (piu_annotate/ml/featurizers.py:19)
  ↓ ArrowDataPoint.to_array_categorical (piu_annotate/ml/datapoints.py:42)
  ↓ cli/limbuse/cache_chunks.py → .npz (x, y, x_mirror, y_mirror)
  ↓ cli/limbuse/train_mlx.py → MLX Transformer (5M o 14.7M params)
  ↓ safetensors → ModelSuite.load(backend='mlx') (piu_annotate/ml/models.py:54)
  ↓ predictor.predict (piu_annotate/ml/predictor.py:13)
  ↓ Tactician beam-search + reglas (piu_annotate/ml/tactics.py:57)
  ↓ Comparator: scripts/compare_models.py:66 + comparations/comparation.html
```

### 1.2 Acoplamiento a hardware (qué bloquea AMD)

| Archivo | Líneas con MLX | Acopla a Apple |
|---|---|---|
| `piu_annotate/ml/mlx_architecture.py` | 1-75 (todo) | Sí — `mlx.core`, `mlx.nn` |
| `piu_annotate/ml/models.py` | 22-23, 70-89, 127-218 | Sí — `MLXModel` class |
| `cli/limbuse/train_mlx.py` | 17-22, 257-674 (la mayoría) | Sí — `mlx.optimizers`, `mx.compile`, `tree_flatten` |
| `cli/limbuse/finetune_mlx.py` | (similar a train_mlx) | Sí |
| `cli/limbuse/infer_v8_batch.py` | (similar) | Sí |
| `piu_annotate/ml/predictor.py` | 0 | **No** — caja-negra a través de `ModelSuite` |
| `piu_annotate/ml/tactics.py` | 0 | **No** — solo numpy + log-probs |
| `piu_annotate/ml/featurizers.py` | 0 | **No** |
| `cli/limbuse/cache_chunks.py` | 0 | **No** — numpy puro |
| `piu_annotate/ml/datapoints.py` | 0 | **No** |
| `piu_annotate/formats/mirror.py` | 0 | **No** |

**Conclusión:** el dataset, la featurización, el cache, el tactician y el predictor son backend-agnostic. Solo arquitectura + loop de entrenamiento + carga de pesos necesitan duplicarse.

### 1.3 Lo que ya funciona bien (no romper)

- Cache .npz por chart (`cache_chunks.py`) — featurización una sola vez, agnóstico.
- Mirror pre-computado (`x`, `y`, `x_mirror`, `y_mirror` en cada .npz) — augmentation gratis.
- Causal mask + prev_limb teacher-forcing — fix de v7b, sigue siendo correcto.
- Tactician con beam-search + reglas duras (impossible-multihit, hold-coherence) — independiente del modelo.

### 1.4 Lo que está mal o muerto

- `MLXModel.predict_prob` (models.py:145) usa `chunk_size=512` mientras `train_mlx.py` usa 1024. **Mismatch.** Las predicciones de borde son peores que en validación.
- `ModelSuite.load` con `backend='mlx'` (models.py:79) **hardcodea** `input_dim = 18` para singles / 23 doubles, pero el modelo actual usa 28 / 33 (con prev_limb one-hot). Esto **rompe** la inferencia MLX desde `ModelSuite`. La ruta v8 funciona porque `infer_v8_batch.py` no usa `ModelSuite` — carga directo con `load_v8_model`.
- `DummyMatchModel` (models.py:46) silenciosamente reemplaza match_next/match_prev cuando `backend='mlx'`. El Tactician sigue recibiendo `0.5` constante → término muerto en `score()`. Antes de mover a producción, **o entrenas los modelos match con MLX o quitas el término del score**.
- `MLX_TRAINING_BUG_REPORT.md` — los bugs ya están fixeados (FIX 1–4 aplicados). Marcar como histórico o borrar.
- Logs `out_*.log` (28 archivos en raíz) — ruido. Mover a `logs/` o `.gitignore`.
- `cli/limbuse/train_lgbm.py` (273 LOC) — superado por MLX. Mantener solo si vas a usarlo como sanity-check baseline; si no, archivar.
- `_flatten_params` artesanal — ya no existe (limpiado).

---

## 2. Bloque A — Backend Dual (MLX + PyTorch/ROCm)

### 2.1 Estrategia

Tres capas. La de arriba habla a la de abajo solo por interfaces.

```
┌─────────────────────────────────────────────────────────────┐
│  predictor.predict + tactics + featurizers + cache (PURO)   │  ← sin cambios
├─────────────────────────────────────────────────────────────┤
│  ModelSuite (backend switch)                                 │  ← refactor pequeño
│  ┌───────────────────┐  ┌──────────────────┐                 │
│  │  MLXModel         │  │  TorchModel      │                 │
│  │  (Apple Silicon)  │  │  (ROCm/CUDA/CPU) │                 │
│  └───────────────────┘  └──────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│  Training loops                                              │
│  ┌───────────────────┐  ┌──────────────────┐                 │
│  │  train_mlx.py     │  │  train_torch.py  │                 │
│  └───────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Contrato del modelo (idéntico en ambos backends)

| Campo | Valor | Razón |
|---|---|---|
| `input_dim` | 28 (singles) / 33 (doubles) | Idéntico al actual; prev_limb one-hot incluido |
| `d_model` | 256 (small) / 384 (large) | v7b / v8 baselines |
| `n_heads` | 8 | sin cambios |
| `n_layers` | 6 (small) / 8 (large) | sin cambios |
| `ffn_dim` | 1024 (small) / 1536 (large) | sin cambios |
| `n_classes` | 3 (L/R/E) | sin cambios |
| `max_len` | 1024 | sin cambios |
| `dropout` | 0.1 | sin cambios |
| **Atención** | Causal, additive mask `-1e9` upper-triangular + pad-mask | sin cambios |
| **Pos enc** | Sinusoidal Vaswani | sin cambios |
| **Weight names** | `input_norm.{weight,bias}`, `input_proj.{weight,bias}`, `encoder.layers.{i}.attention.{...}`, `out_head.fc1.{weight,bias}`, etc. | **CLAVE**: nombres alineados habilitan conversión MLX↔PyTorch sin código manual |

### 2.3 Nuevo: `piu_annotate/ml/arch_torch.py`

Espejo 1:1 de `mlx_architecture.py`. Reemplazar `mlx.core`→`torch`, `mlx.nn`→`torch.nn`. Mismo número de parámetros, mismas dimensiones, mismos nombres.

Skeleton (no copies literal, escríbelo correctamente):

```python
from __future__ import annotations
import math
import torch
import torch.nn as nn


def sinusoidal_pos_encoding(seq_len: int, d_model: int, device=None) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term[:pe[:, 1::2].shape[1]])
    return pe


class OutputHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class LimbSequenceTransformerTorch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        max_len: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.register_buffer('pos_enc', sinusoidal_pos_encoding(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = OutputHead(d_model, n_classes, dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D_in)  padding_mask: (B, L) bool, True = padding
        B, L, _ = x.shape
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = h + self.pos_enc[:L].unsqueeze(0)
        h = self.dropout(h)
        # Causal mask: float -inf upper triangular
        causal = torch.zeros(L, L, device=x.device)
        causal.masked_fill_(torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1), float('-inf'))
        h = self.encoder(h, mask=causal, src_key_padding_mask=padding_mask)
        h = self.out_norm(h)
        return self.out_head(h)
```

**Atención: nombres de pesos.** `nn.TransformerEncoderLayer` de PyTorch usa nombres `self_attn.in_proj_weight`, `self_attn.out_proj.weight`, `linear1.weight`, `linear2.weight`. MLX usa otros. Para hacer interoperables los `.safetensors` necesitas **wrapper manual con sub-módulos nombrados igual que MLX**. Si quieres pesos cross-backend, escribe el encoder a mano (≈100 LOC) con `nn.MultiheadAttention`, `nn.Linear`, `nn.LayerNorm` con los mismos paths que MLX.

**Recomendación:** **NO** intentes compartir pesos cross-backend en la primera iteración. Entrena cada backend por separado, comparte solo el contrato. Después, si lo necesitas, escribe `scripts/convert_weights_mlx_torch.py` que renombre claves.

### 2.4 Nuevo: `cli/limbuse/train_torch.py`

Estructura idéntica a `train_mlx.py`. Cambios mecánicos:

| MLX | PyTorch |
|---|---|
| `mx.array(np_arr)` | `torch.from_numpy(np_arr).to(device)` |
| `mx.eval(...)` | innecesario (PyTorch es eager por defecto) |
| `mx.compile(fn)` | `torch.compile(fn)` (PyTorch 2.x) |
| `nn.value_and_grad` | `loss.backward()` + `optimizer.step()` |
| `mlx.optimizers.AdamW` | `torch.optim.AdamW` |
| `mx.save_safetensors` | `safetensors.torch.save_file` |
| `mx_clip_grad_norm` | `torch.nn.utils.clip_grad_norm_` |
| `linear_schedule + cosine_decay` | `torch.optim.lr_scheduler.OneCycleLR` o manual |

Device pickup:
```python
def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')  # ROCm también expone CUDA API
    if torch.backends.mps.is_available():
        return torch.device('mps')   # Apple Silicon fallback
    return torch.device('cpu')
```

**Para AMD 6950 XT (gfx1030, RDNA2):**

```bash
# Instalación ROCm PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
# Verificar
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

ROCm presenta el ROCm device como `cuda` en la API de PyTorch (mismas calls). El `HSA_OVERRIDE_GFX_VERSION=10.3.0` es necesario porque oficialmente PyTorch ROCm sólo soporta gfx1030 con override de versión (RDNA2 navi21).

### 2.5 Refactor `models.py`: switch limpio

Actual `ModelSuite.__init__` (line 54) hardcodea por `model_type`. Limpiar:

```python
class ModelSuite:
    def __init__(self, singles_or_doubles: str, backend: str = None):
        self.singles_or_doubles = singles_or_doubles
        # backend ∈ {'lightgbm', 'mlx', 'torch'}; default: hargs['model']
        self.backend = backend or args['model']
        loader = {
            'lightgbm': self._load_lgbm,
            'mlx':      self._load_mlx,
            'torch':    self._load_torch,
        }[self.backend]
        sd = singles_or_doubles
        self.model_arrows_to_limb       = loader(f'arrows_to_limb-{sd}')
        self.model_arrowlimbs_to_limb   = loader(f'arrowlimbs_to_limb-{sd}', fallback='arrows_to_limb')
        self.model_arrows_to_matchnext  = self._maybe_dummy(loader, f'arrows_to_matchnext-{sd}')
        self.model_arrows_to_matchprev  = self._maybe_dummy(loader, f'arrows_to_matchprev-{sd}')
```

**Fix obligatorio:** sustituir el `input_dim = 18/23` hardcoded de `models.py:79-82` por lectura del `.meta` JSON guardado por `train_mlx.py:600-608`. Es la única fuente de verdad.

```python
def _load_mlx(self, key: str, fallback: str | None = None) -> ModelWrapper:
    sd = self.singles_or_doubles
    model_file = os.path.join(args['model.dir'], f"{sd}-{key.split('-')[0]}-mlx-best.safetensors")
    meta_file  = model_file.replace('.safetensors', '.meta')
    if not os.path.exists(model_file) and fallback:
        return self._load_mlx(f'{fallback}-{sd}')
    with open(meta_file) as f:
        meta = json.load(f)
    return MLXModel.load(model_file, **meta)
```

Y `MLXModel.load` acepta `**meta` completo (input_dim, d_model, n_heads, n_layers, ffn_dim, n_classes). Lo mismo para `TorchModel.load`.

### 2.6 Inferencia: alinear chunk_size con training

`models.py:147` usa `chunk_size=512`, `train_mlx.py:27` usa `MAX_SEQ_LEN=1024`. Subir a 1024 (los charts mas largos del val set tienen <1024 tokens en >99%). Si OOM, mantener 512 pero **documentar el AR-gap** que esto introduce.

---

## 3. Bloque B — Two-Pass Limb (coarse → refine)

### 3.1 Motivación

Tu pregunta original: "predict patter in ssc but also predict wich fot are gonna be pressed are two steps". Lectura: **dos modelos en cascada**, donde el segundo refina al primero.

Hoy el modelo único produce logits → argmax → tactician beam-search. El refinamiento ocurre **fuera del modelo** (en tactics.py). Pasar lógica de refinamiento adentro del modelo aumenta el techo porque:

1. El refinador ve el patrón completo del pase 1, no sólo prev_limb.
2. Permite condicionar en estructuras macro (jacks-en-streak, brackets-en-burst).
3. Reduce dependencia del Tactician (más simple post-proc).

### 3.2 Diseño

```
Pase 1 (coarse): arrow features → logits L/R/E   [igual al modelo actual]
   ↓ argmax
Pase 2 (refine): arrow features + pase-1 logits   → logits L/R/E corregidos
   ↓ argmax
   ↓ tactician (reducido: solo enforcement de reglas duras)
   ↓ output final
```

**Implementación:** dos modelos `LimbSequenceTransformer` con el **mismo backbone** (compartir pesos del encoder es una opción) pero distinto input.

| Modelo | Input dim | Notas |
|---|---|---|
| `arrows_to_limb` (coarse) | 28/33 | Idéntico al actual |
| `arrowlimbs_to_limb` (refine) | 28+3 = 31 / 33+3 = 36 | Concatena logits soft de pase 1 al feature vector |

El nombre `arrowlimbs_to_limb` **ya existe** en el suite LGBM (`models.py:63`). Hoy es dummy para MLX. Convertirlo en el refinador es backwards-compatible con `ModelSuite`.

### 3.3 Procedimiento de entrenamiento

1. **Stage 1: train coarse.** Identico al actual. `cli/limbuse/train_{mlx,torch}.py --task arrows_to_limb`. Salida: `{sd}-arrows_to_limb-{backend}-best.safetensors`.
2. **Stage 2: train refine.** Nuevo script `cli/limbuse/train_refine_{mlx,torch}.py`. Para cada batch:
   - Forward pase 1 (modelo coarse, frozen). Toma soft-max → 3 floats por token.
   - Concat al feature vector. Forward pase 2 (modelo refine, trainable).
   - CE loss vs ground truth.
3. Mirror augmentation y prev_limb teacher-forcing se mantienen igual.
4. Scheduled-sampling **se elimina** — el refinador ya ve los errores del pase 1 (es más realista que SS).

**Ahorro vs scheduled-sampling actual:** el refinador es entrenable con el coarse congelado → grads sólo fluyen por refine → ~40% más rápido por step que SS bi-pase actual.

### 3.4 Cabeza alternativa (más ambiciosa)

En vez de dos modelos separados, **un modelo con dos cabezas + iteración interna**:

```python
class TwoPassTransformer(nn.Module):
    def forward(self, x, pm):
        h1 = self.encoder1(x, pm)
        logits1 = self.head_coarse(h1)        # (B, L, 3)
        # Concat soft pase-1 al input y re-encodea
        x2 = torch.cat([x, softmax(logits1, dim=-1)], dim=-1)
        h2 = self.encoder2(x2, pm)
        logits2 = self.head_refine(h2)
        return logits1, logits2               # loss combinada λ·CE(logits1) + (1-λ)·CE(logits2)
```

Pros: una sola pasada por el grafo de cómputo, gradients fluyen end-to-end.
Contras: encoder2 es nuevo (más params), o re-usas encoder1 (menos capacidad pero más rápido).

**Recomendación pragmática:** Stage A (sec. 3.3) primero. Si no llega a 93%+ singles, considera Stage B (one-model two-head).

### 3.5 Métricas para validar B

| Métrica | v8 actual | Target two-pass |
|---|---|---|
| Singles val_acc | 91.4% | **≥ 93%** |
| Singles AR val_acc | (medir) | ≤ 1pp gap vs teacher-forcing |
| Brackets RR acc | (medir) | ≥ Brackets LL acc - 1pp |
| Triple-taps | 86.9% (LGBM) | ≥ 92% |
| Inference time/chart | ~baseline | ≤ 1.5× baseline (pase 2 añade ~40% cómputo) |

Si pase 2 sube val_acc < 0.5pp, **mata la idea** — no vale la complejidad.

---

## 4. Bloque C — Comparador 2.0

### 4.1 Estado actual

| Componente | Archivo | Qué hace |
|---|---|---|
| Script Python | `scripts/compare_models.py:66` | Score tap/jack/triple sobre 30 worst + 30 hardest |
| Viewer browser | `comparations/comparation.html` | Konva canvas, sidebar charts, sin diff jump |
| Reportes HTML | `scripts/generate_v8_comparison_report.py` etc. | One-off por modelo, no comparables entre sí |

### 4.2 Mejoras solicitadas (priorizadas)

#### C1 — Per-pattern accuracy desglosado

En `scripts/compare_models.py`, ampliar `score_limbs_vs_ref` para clasificar cada predicción en buckets físicos antes de contar:

```python
PATTERN_TYPES = ['tap', 'jack', 'triple', 'bracket_ll', 'bracket_rr', 'bracket_lr', 'jump', 'hold_release', 'stream']
```

Cómo clasificar (ya está la mayoría en `tactics.py` y `featurizers.py`):

| Bucket | Definición |
|---|---|
| `tap` | `num_downpress_in_line == 1` |
| `jack` | tap con `n_same_panel_streak >= 2` |
| `triple` | `num_downpress_in_line == 3` |
| `bracket_ll` | `num_downpress_in_line == 2`, `line_is_bracketable`, gt = `ll` |
| `bracket_rr` | mismo, gt = `rr` |
| `bracket_lr` / `rl` | jump bilateral |
| `jump` | `num_downpress_in_line == 2`, no bracketable |
| `hold_release` | `prior_line_only_releases_hold_on_this_arrow == True` |
| `stream` | tap en streak con `time_since_prev_downpress < 0.12s` |

Output: nueva tabla con columnas `[pattern, n, model_a_acc, model_b_acc, delta_pp]`.

#### C2 — Confusion matrix L/R/E + per-difficulty plot

Añadir a `compare_models.py` un modo `--confusion`:

```python
def confusion_matrix_3x3(preds, gt, mask):
    cm = np.zeros((3, 3), dtype=int)
    for p, g, m in zip(preds, gt, mask):
        if m: cm[g, p] += 1
    return cm  # rows=truth, cols=pred
```

Plot:
- Heatmap 3x3 normalizado por fila (recall L/R/E).
- Bar chart accuracy por nivel (S1..S26, D1..D28) — ya está la columna `level` en `baseline`.

Guardar en `artifacts/comparator/confusion_{model}.png`.

#### C3 — Diff viewer con timestamp seek (multi-model)

Reescribir `comparations/comparation.html` (ver sección 4.3 abajo).

#### C4 — Soporte multi-model en `compare_models.py`

Actualmente compara `old (processed_db)` vs `new (re-inference con suite)`. Generalizar a N modelos vía flag:

```bash
python scripts/compare_models.py \
  --models lgbm:artifacts/models/visss \
  --models mlx_v8:artifacts/models/visss-mlx-v8 \
  --models torch_rocm:artifacts/models/visss-torch-rocm \
  --baseline artifacts/benchmark_baseline_60.json \
  --output_json artifacts/comparator/multi_compare.json
```

Output JSON consumido por el viewer (sec. 4.3).

### 4.3 Diff Viewer 2.0 — `comparations/comparation_v2.html`

**Decisión:** no romper `comparation.html` actual. Crear `comparation_v2.html` paralelo y deprecar el viejo después.

**Cambios clave:**

| Feature | Cómo |
|---|---|
| **Multi-model panes** | Grid CSS `1fr 1fr 1fr` para 3 modelos lado-a-lado en vez de 2. Lista de modelos dinámica desde `index.json` |
| **Diff list con jump** | Sidebar segundo nivel: lista de errores (rows donde algún modelo diverge de gt). Click → scroll a esa fila + highlight |
| **Timestamp seek** | Si hay video/audio: input `<input type=number>` para tiempo en segundos → calcula `row_idx` por interpolación y scrollea |
| **Per-pattern filter** | Dropdown: `[all, taps, jacks, triples, brackets-rr, brackets-ll, holds]`. Filtra qué errores muestra |
| **Confusion mini-widget** | Esquina superior derecha: 3×3 grid mostrando counts L/R/E del chart actual |
| **Audio sync (futuro)** | `<audio>` con `currentTime` linkeado a scroll del canvas. Para cuando esté integrado con music-to-steps |

Estructura JS limpia:

```html
<script>
  // 1. Carga index.json — lista de [{shortname, songname, models: {a, b, c}, ref}]
  // 2. Para cada chart, fetch model.json y ref.json al seleccionar
  // 3. Diff engine: row-by-row compare → list of {row_idx, t, gt_limb, models: {a: 'l', b: 'r', c: 'l'}}
  // 4. Render con Konva en N paneles sincronizados scroll-wise
  // 5. Click en diff item → scrollAll(row_idx); highlight ring on that arrow in all panes
</script>
```

Mantén Konva (ya está cargado). No metas React.

### 4.4 Pipeline de generación de reportes

Hoy hay 3 scripts: `generate_v8_comparison_report.py`, `_simple.py`, `_v9_ft_*`. Unificar:

```bash
scripts/generate_comparison_report.py \
  --models <name:dir> [<name:dir>...] \
  --baseline artifacts/benchmark_baseline_60.json \
  --out_html artifacts/comparator/report.html \
  --out_json artifacts/comparator/data.json \
  --include confusion per_pattern per_level
```

Generar **un solo HTML autocontenido** + JSON con todos los números. El JSON alimenta también `comparation_v2.html`.

Borrar después: `generate_v8_comparison_*` (3 archivos), `new_architecture_report.html`, `v8_model_comparison_report.html`, `v9_ft_model_comparison_report.html`.

---

## 5. Plan de migración (orden recomendado)

### Fase 0 — Limpieza (30 min, sin riesgo)

1. Mover `out_*.log` (28 archivos) a `logs/` y añadir `logs/` a `.gitignore`.
2. Marcar `MLX_TRAINING_BUG_REPORT.md` como histórico (mover a `docs/historic/`).
3. Decidir qué hacer con `train_lgbm.py` (mantener como sanity-check vs archivar).
4. Fijar `MLXModel` `input_dim` desde `.meta` (sec. 2.5) — fix de bug bloqueante.
5. Subir `chunk_size` de inferencia MLX de 512 a 1024 (sec. 2.6).

### Fase 1 — Backend PyTorch (4–6 h)

1. Crear `piu_annotate/ml/arch_torch.py` (sec. 2.3). Smoke con random input.
2. Crear `cli/limbuse/train_torch.py` (sec. 2.4). Importar todo de `train_mlx.py` (build_batches, make_chunks, compute_loss numpy version, save_safetensors).
3. Crear `piu_annotate/ml/models_torch.py` con `TorchModel(ModelWrapper)`. Mismo interface que `MLXModel`.
4. Refactor `ModelSuite` con switch limpio (sec. 2.5).
5. Test: entrenar 50 charts × 2 epochs en MPS (Mac) → loss baja, `.safetensors` carga. **Aún no toques 6950 XT.**
6. Test: mismo smoke en CPU `--device cpu`. Reproduce loss bit-by-bit (no, no es bit-by-bit pero similar).
7. **Solo entonces:** ir a la máquina con 6950 XT, instalar ROCm PyTorch (sec. 2.4), correr el mismo smoke.

### Fase 2 — Two-pass refine (3–5 h)

1. Confirmar que el modelo coarse actual entrena en ambos backends y produce val_acc consistente.
2. Crear `cli/limbuse/train_refine_mlx.py` (sec. 3.3). El modelo coarse se carga frozen.
3. Smoke: 50 charts × 2 epochs. Verificar que el refinador converge.
4. Entrenar refine full sobre singles con el coarse v8 congelado. Evaluar contra v8 puro.
5. Si val_acc sube ≥ 1pp: replicar en PyTorch + doubles.

### Fase 3 — Comparador 2.0 (4–6 h)

1. `compare_models.py`: extender `score_limbs_vs_ref` con buckets (sec. 4.2 C1). Añadir `--confusion` (C2). Añadir multi-model (C4).
2. Unificar scripts de reportes → `generate_comparison_report.py` (sec. 4.4).
3. Crear `comparation_v2.html` (sec. 4.3). Mantener viejo paralelo hasta validar.
4. Generar reporte master con todos los modelos disponibles. Subir a piulatam si es informativo.

### Fase 4 — Música → pasos (futuro)

Diseño separado (sec. 6).

---

## 6. Apéndice — Música → pasos (forward-looking, no implementar aún)

Tu objetivo a futuro: predecir notas desde audio musical. La idea es que el modelo limb actual sea **componente reutilizable** del pipeline final.

### 6.1 Pipeline propuesto

```
audio (.ogg / .wav)
  ↓ librosa mel-spectrogram (n_mels=128, hop=160, sr=16000)
  ↓ Music Encoder Transformer (input=mel-frames, output=audio embeddings @ ~100Hz)
  ↓ Note Decoder: cross-attention al audio + autoregresivo en notas
  ↓ sequence de note-events (panel, time, type[tap/hold_head/hold_tail])
  ↓ Convertidor SSC → ChartStruct CSV
  ↓ ARROW DECODER (== modelo limb actual): chart features → limb labels
  ↓ output: chart anotado completo
```

Tres bloques entrenables:

| Bloque | Datos | Loss | Reutiliza |
|---|---|---|---|
| Music Encoder | (audio, chart) pares | Frame-level note presence + onset | Pre-train con datasets de música general (ej. NSynth) |
| Note Decoder | (audio emb, notes) pares | Token-CE sobre vocabulario [panel × type] | — |
| Limb Decoder | (notes, limbs) pares = **dataset actual** | CE 3-class | **El modelo actual** |

### 6.2 Por qué la arquitectura two-pass de Bloque B aporta acá

El refinador (`arrowlimbs_to_limb`) que vas a entrenar tiene **input de logits soft del coarse**. Lo mismo aplica al pipeline música: el Limb Decoder puede recibir **logits soft del Note Decoder** como input, no solo notas hard. Esto da:

- Robustez a ambigüedad de note decoder (si dice "60% panel 2, 40% panel 0", el limb decoder lo sabe).
- Training joint posible: backprop end-to-end audio → limb sobre charts anotados con audio.

**Acción concreta para mantener compatibilidad futura:** asegúrate de que el feature vector del refinador acepte **probas float** en lugar de **labels hard one-hot**. Ya es lo natural (softmax del coarse) — solo no hardcodees `one_hot(argmax)` en ningún sitio.

### 6.3 Datasets sugeridos para fase música

- StepMania archives con audio: ITG, ECS, etc. (~1000s de canciones con SSC).
- Pump It Up Phoenix data (si tienes acceso). Tu `visss-120524` ya tiene shortname → buscar audio en el servidor piumx.
- Augmentation: pitch-shift ±2 semitones, time-stretch ±5%.

### 6.4 No-implementar-ahora

Demasiado scope. Documentado para que Bloque B se diseñe pensando en esta extensión, nada más.

---

## 7. Tabla resumen de archivos

### Nuevos

| Archivo | Líneas est. | Bloque |
|---|---|---|
| `piu_annotate/ml/arch_torch.py` | ~100 | A |
| `piu_annotate/ml/models_torch.py` | ~120 | A |
| `cli/limbuse/train_torch.py` | ~600 (port de train_mlx) | A |
| `cli/limbuse/train_refine_mlx.py` | ~400 | B |
| `cli/limbuse/train_refine_torch.py` | ~400 | B |
| `scripts/generate_comparison_report.py` | ~300 | C |
| `comparations/comparation_v2.html` | ~600 | C |
| `scripts/convert_weights_mlx_torch.py` | ~50 (opcional) | A |

### Modificados

| Archivo | Cambio | Bloque |
|---|---|---|
| `piu_annotate/ml/models.py` | Switch backend limpio, fix input_dim desde .meta | A |
| `piu_annotate/ml/mlx_architecture.py` | Sin cambios funcionales; alinear nombres de pesos si vas por cross-backend | A |
| `scripts/compare_models.py` | Per-pattern + confusion + multi-model | C |

### Archivados / borrados

| Archivo | Razón |
|---|---|
| `MLX_TRAINING_BUG_REPORT.md` | Bugs fixeados, mover a `docs/historic/` |
| `out_*.log` (28) | Mover a `logs/` + `.gitignore` |
| `new_architecture_report.html` | Stale; reemplazar por reporte unificado |
| `v8_model_comparison_report.html` | Stale; reemplazar por reporte unificado |
| `v9_ft_model_comparison_report.html` | Stale; reemplazar por reporte unificado |
| `scripts/generate_v8_comparison_report.py` | Reemplazado por `generate_comparison_report.py` |
| `scripts/generate_v8_comparison_simple.py` | Reemplazado |
| `scripts/generate_v9_ft_comparison_report.py` | Reemplazado |
| `cli/limbuse/train_lgbm.py` | Mantener si quieres baseline; archivar si no |
| `out.log`, `smoke_test.log`, `smoke_test_v9.log` | Logs de runs sueltos |
| `benchmark_v3_ignore_e.log`, `benchmark_v4_relaxed_e.log` | Históricos |

---

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| ROCm + PyTorch en 6950 XT crashea (gfx1030 no oficial) | Tener `--device cpu` y `--device mps` como fallback en `train_torch.py`. Smoke siempre primero |
| Two-pass refine no mejora val_acc | Mata Bloque B, mantén v8 puro. Bloque A y C tienen valor independiente |
| Conversion de pesos MLX↔PyTorch falla | Documentado que no es objetivo inicial. Entrenar separados |
| Comparador v2 rompe workflows | Mantener `comparation.html` paralelo hasta validar |
| Tactician falla con backend torch | Tactician es backend-agnostic (recibe log-probs como numpy). Verificar con prints en smoke |
| `DummyMatchModel` distorsiona score en MLX/torch | O entrenar match_next/match_prev en cada backend, o quitar el término del score cuando match models son dummy. Decidir antes de Fase 2 |

---

## 9. Métricas de éxito globales

Antes de cerrar el proyecto, verificar que **todas** se cumplen:

- [ ] Backend MLX entrena en M5 Pro a velocidad ≥ baseline actual.
- [ ] Backend PyTorch entrena en 6950 XT (ROCm) sin crash, smoke completa.
- [ ] Backend PyTorch en M5 Pro (MPS) entrena pero documentado como ~30% más lento que MLX.
- [ ] Mismo modelo (coarse) entrenado en MLX y en PyTorch (ROCm) llega a val_acc dentro de ±0.5pp.
- [ ] Two-pass refine (si se completa Bloque B) llega a singles val_acc ≥ 93%.
- [ ] Comparator nuevo muestra per-pattern + confusion matrix correctamente.
- [ ] Diff viewer v2 carga 3 modelos lado a lado y permite jump-to-error.
- [ ] `ModelSuite(backend='mlx')` carga sin hardcode de input_dim (fix de bug).

---

## 10. Decisiones que el agente debe tomar (no asumir)

Ante cualquiera de estas, **detente y pregunta** en vez de improvisar:

1. Si los pesos MLX no son convertibles a PyTorch sin escribir encoder manual: ¿escribir encoder manual o aceptar que cada backend entrena desde cero?
2. Si ROCm PyTorch crashea en 6950 XT con gfx1030: ¿downgrade a PyTorch 2.4 o usar Vulkan compute (ncnn) como plan B?
3. Si two-pass refine baja inference speed >2×: ¿aceptar o reducir el modelo refine?
4. Si comparator v2 necesita backend WebGL para >1000 notes: ¿quedarse con Konva o migrar a regl/three?
5. Si el dataset actual es insuficiente para refine (overfitting): ¿juntar otro corpus o quedarse en stage 1?

---

## 11. Cierre

Tres bloques. Independientes. El usuario corre los entrenamientos cuando quiera, no es responsabilidad de este documento ni del agente que lo implemente.

**Orden recomendado:** A (limpieza + backend dual) → C (comparator) → B (two-pass) → futuro (música).

**Por qué este orden:** A desbloquea AMD (capacidad de cómputo extra para el futuro). C permite medir bien lo que se entrene en A y B. B es el salto de techo, pero solo vale si C lo puede medir bien.

**Una cosa más.** El campo `n_classes=3` en `mlx_architecture.py:39` (L/R/E) está bien para limb, pero cuando llegues a Música→pasos cambia: vocabulario es mucho más grande (panel × event-type). No re-uses la misma cabeza — instancia un Transformer separado por tarea.

Fin del documento.
