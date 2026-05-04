# MLX Transformer Training — Guía Optimizada para Continuar sin Perder Funcionalidades

**Fecha:** 2026-05-04
**Estado del proyecto:** El intento anterior (PID 92111, 2026-05-04) falló silenciosamente tras ~18 min con OOM probable. Esta guía aplica los fixes de los 5 bugs documentados en `MLX_TRAINING_BUG_REPORT.md` y añade optimizaciones para que el siguiente run sea **significativamente más rápido** (objetivo: <15 min para singles, ~30 min para doubles) y **reproducible** (modelo cargable + pipeline cacheado).

**Hardware asumido:** Apple Silicon (M-series), 16–24 GB RAM unificada.
**MLX version:** 0.29.3 (verificada).
**Target final:** ≥ 90% val_acc en singles (vs 75.4% LightGBM).

---

## 0. Filosofía de la migración

El plan no es "rehacer todo lo del LightGBM". Es:
1. **Mantener** el pipeline LGBM intacto como fallback en producción (nada se rompe si MLX falla otra vez).
2. **Aislar** el experimento MLX en `artifacts/models/visss-mlx/` con su propio `--out_dir`.
3. **Validar** rápido con un smoke test (50 charts, 1 epoch) antes de quemar 30 min de training.
4. **Medir** memoria y guardar checkpoints para que un crash no borre 20 epochs de trabajo.

> Regla: **si el smoke test no termina exitosamente y produce un archivo cargable, NO lances el training completo.**

---

## 1. Fixes obligatorios (block training)

Estos 4 fixes deben aplicarse antes de cualquier run. Cada uno es un patch específico — no toques nada más.

### FIX 1 — Save format (BUG 1, HIGH)

**Síntoma actual:** `np.savez(path.safetensors, ...)` produce un `.npz` con extensión engañosa. `mx.load()` falla. Ningún run anterior produjo un modelo cargable.

**Archivo:** `cli/limbuse/train_mlx.py`

**Reemplazar las dos zonas de save (líneas ~299–310):**

```python
# ANTES
save_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
state = model.parameters()
flat_state = _flatten_params(state)
np.savez(save_path, **flat_state)
# ...
final_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-final.safetensors')
flat_state = _flatten_params(model.parameters())
np.savez(final_path, **flat_state)
```

```python
# DESPUÉS — usa el API nativo de MLX
from mlx.utils import tree_flatten

def save_model(path: str, model: nn.Module) -> None:
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(path, flat)

# en train(...):
save_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-best.safetensors')
save_model(save_path, model)
# ...
final_path = os.path.join(out_dir, f'{sd}-arrows_to_limb-mlx-final.safetensors')
save_model(final_path, model)
```

**Por qué `tree_flatten` y no `_flatten_params` artesanal:** `tree_flatten` es la utility oficial de MLX, maneja listas/dicts/módulos correctamente y produce las dot-keys que `tree_unflatten` espera para cargar.

**Borrar:** la función `_flatten_params` ya no se usa. Eliminala (líneas 35–46).

**Verificación:**
```bash
python3 -c "
import mlx.core as mx
d = mx.load('artifacts/models/visss-mlx/singles-arrows_to_limb-mlx-best.safetensors')
print(f'OK — {len(d)} tensors loaded')
print(list(d.keys())[:5])
"
```

---

### FIX 2 — Mirror augmentation (BUG 2, HIGH)

**Síntoma actual:** Cuando se aplica mirror, el código sobrescribe `cx[:min_len] = mx2[:min_len]` y `cy[:min_len] = my[:min_len]`. Si `min_len < len(cx)`, el resto del chunk queda con features originales pero la **mitad de los labels** son del chart espejado → datos corruptos.

**Archivo:** `cli/limbuse/train_mlx.py`, líneas 136–148.

```python
# ANTES
for cx, cy in chunks:
    was_mirrored = False
    if mirror_prob > 0 and rng.random() < mirror_prob:
        mirrored_cs = mirror_chartstruct(cs)
        mx2, my = chartstruct_to_sequence(mirrored_cs)
        min_len = min(len(cx), len(mx2))
        cx[:min_len] = mx2[:min_len]
        cy[:min_len] = my[:min_len]
        was_mirrored = True
    counts['total'] += 1
    if was_mirrored:
        counts['mirrored'] += 1
    all_chunks.append((cx, cy))
```

```python
# DESPUÉS — mirror al nivel de CHART, no de chunk
should_mirror = mirror_prob > 0 and rng.random() < mirror_prob
if should_mirror:
    cs_used = mirror_chartstruct(cs)
    x, y = chartstruct_to_sequence(cs_used)
    chunks = make_chunks(x, y)
    counts['mirrored'] += len(chunks)
counts['total'] += len(chunks)
for cx, cy in chunks:
    all_chunks.append((cx, cy))
```

**Reorganización del loop completo** (la decisión de mirror se mueve antes de `chunks = make_chunks(...)`):

```python
for f in tqdm(files, desc=f'Loading {sd}'):
    try:
        cs = ChartStruct.from_file(str(f))
        if cs.singles_or_doubles() != sd:
            continue
        should_mirror = mirror_prob > 0 and rng.random() < mirror_prob
        cs_used = mirror_chartstruct(cs) if should_mirror else cs
        x, y = chartstruct_to_sequence(cs_used)
        chunks = make_chunks(x, y)
        counts['total'] += len(chunks)
        if should_mirror:
            counts['mirrored'] += len(chunks)
        for cx, cy in chunks:
            all_chunks.append((cx, cy))
    except Exception as e:
        logger.warning(f'Error loading {f}: {e}')
```

**Por qué a nivel de chart y no de chunk:** la augmentación no debe partir un chart en chunks "mitad mirror, mitad no" — pierde coherencia temporal. Y si vas a aplicarla siempre al chart entero, hacerlo antes del chunking es lo correcto.

---

### FIX 3 — Attention mask shape (BUG 3, MEDIUM)

**Archivo:** `piu_annotate/ml/mlx_architecture.py:52–55` **y** `cli/limbuse/train_mlx.py:89–92` (están duplicados — usa solo el de `mlx_architecture.py`).

```python
# ANTES
if mx.sum(padding_mask) > 0:
    attn_mask = mx.where(padding_mask[:, None, :, None], -1e9, 0.0)  # (B, 1, L, 1) — NO broadcastea
else:
    attn_mask = None
```

```python
# DESPUÉS
attn_mask = mx.where(padding_mask[:, None, :], -1e9, 0.0)  # (B, 1, L) → broadcastea a (B, H, L, L)
```

**Quita el guard `if mx.sum > 0`:** en batches de longitud variable casi siempre hay padding, el guard solo añade un sync y una rama.

**Y borra la versión duplicada en `train_mlx.py:84–94`:** la clase `LimbSequenceTransformer` ya está definida en `mlx_architecture.py`. En `train_mlx.py` solo importa:

```python
from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer
```

(Quita la definición duplicada y el `_build_padding_mask` muerto.)

---

### FIX 4 — Lazy loading + cache a disco (BUG 4, MEDIUM → optimización mayor)

**Síntoma actual:** los 4261 charts se featurizan en cada run desde cero (~ 5 min) y se materializan todos los chunks en RAM antes de empezar (~5.5 GB).

**Solución:** featurizar **una sola vez**, cachear a disco como `.npz` por chart, y al entrenar leer solo lo necesario.

**Crear `cli/limbuse/cache_chunks.py`:**

```python
#!/usr/bin/env python3
"""Pre-featurize chartstructs and cache as .npz per chart. Run once."""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.mirror import mirror_chartstruct
from piu_annotate.ml.featurizers import ChartStructFeaturizer


def featurize(cs):
    ft = ChartStructFeaturizer(cs)
    x_raw = np.stack(ft.pt_array)
    cmf = np.tile(ft.chart_metadata_features, (len(x_raw), 1))
    x = np.concatenate([x_raw, cmf], axis=1).astype(np.float32)
    y = ft.get_labels_from_limb_col('Limb annotation').astype(np.int8)
    return x, y


def main(folder: str, out_dir: str, sd: str):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(Path(folder).glob('*.csv'))
    skipped = 0
    for f in tqdm(files, desc=f'Caching {sd}'):
        try:
            cs = ChartStruct.from_file(str(f))
            if cs.singles_or_doubles() != sd:
                continue
            x, y = featurize(cs)
            xm, ym = featurize(mirror_chartstruct(cs))
            np.savez(
                os.path.join(out_dir, f'{f.stem}.npz'),
                x=x, y=y, x_mirror=xm, y_mirror=ym,
            )
        except Exception as e:
            skipped += 1
    print(f'Done. Skipped {skipped} charts.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sd', required=True, choices=['singles', 'doubles'])
    args = ap.parse_args()
    main(args.folder, args.out_dir, args.sd)
```

**Run UNA VEZ (~3–5 min):**
```bash
python3 cli/limbuse/cache_chunks.py \
  --folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --out_dir artifacts/cache/mlx-singles/ --sd singles
python3 cli/limbuse/cache_chunks.py \
  --folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --out_dir artifacts/cache/mlx-doubles/ --sd doubles
```

**Ahorro:** runs subsecuentes leen `.npz` directamente. De ~5 min a ~10 s. Y el mirror está pre-computado, así que el coin-flip en training solo elige `x` vs `x_mirror` (cero CPU).

**Reemplazar `load_all_chunks` en `train_mlx.py`:**

```python
def load_all_chunks_cached(cache_dir: str, mirror_prob: float, seed: int, limit: int | None = None):
    files = sorted(Path(cache_dir).glob('*.npz'))
    if limit:
        files = files[:limit]
    rng = np.random.default_rng(seed)
    all_chunks = []
    n_mirror = 0
    for f in tqdm(files, desc=f'Loading cached'):
        d = np.load(f)
        if mirror_prob > 0 and rng.random() < mirror_prob:
            x, y = d['x_mirror'], d['y_mirror']
            n_mirror += 1
        else:
            x, y = d['x'], d['y']
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        for cx, cy in make_chunks(x, y):
            all_chunks.append((cx, cy))
    logger.info(f'Loaded {len(all_chunks)} chunks ({n_mirror}/{len(files)} mirrored)')
    return all_chunks
```

---

## 2. Optimizaciones de velocidad (no son fixes — son speedups)

### OPT 1 — Length bucketing (reduce padding desperdiciado)

**Problema:** `build_batches` shufflea aleatorio, así que un batch puede tener 1 chunk de 1024 + 15 de 200 → 14× más padding que cómputo útil.

**Fix:** ordena chunks por longitud antes de batchear, mezcla solo el orden de batches.

```python
def build_batches(chunks, batch_size, shuffle=True, seed=0):
    # Sort once by length (caps padding waste)
    sorted_idx = sorted(range(len(chunks)), key=lambda i: chunks[i][0].shape[0])
    batches = [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    for batch_idxs in batches:
        max_len = max(chunks[i][0].shape[0] for i in batch_idxs)
        # ... resto igual
```

**Ahorro estimado:** 1.5–3× más rápido en epochs (depende de la varianza de longitudes).

---

### OPT 2 — `mx.compile` el forward+loss

```python
# En train_epoch, antes del loop
@mx.compile
def step(model, x, y, pm, lm):
    return compute_loss(model(x, pm), y, lm)

loss_and_grad = nn.value_and_grad(model, step)
```

**Ahorro estimado:** 1.3–1.8× en steps consecutivos. La primera batch compila (lenta), las demás vuelan.

---

### OPT 3 — Eval menos frecuente

`mx.eval(model.parameters(), optimizer.state)` después de **cada** batch fuerza un sync con la GPU. Es overkill — MLX ya evalúa cuando lo necesita para el siguiente forward.

```python
# ANTES (línea 244)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)  # Cada step

# DESPUÉS — eval solo cada N steps
optimizer.update(model, grads)
if n_batches % 8 == 0:
    mx.eval(model.parameters(), optimizer.state)
```

**Cuidado:** al final del epoch, llama `mx.eval(...)` una vez antes de evaluate.

**Ahorro estimado:** 1.2–1.5× en throughput.

---

### OPT 4 — Train/val split sin doble carga

**Problema actual:**
```python
train_chunks = load_all_chunks(folder, sd, mirror_prob=0.5, seed=seed, limit=...)
val_chunks   = load_all_chunks(folder, sd, mirror_prob=0.0, seed=seed+1, limit=...)
```
Esto carga **TODOS** los charts dos veces (uno con mirror, otro sin). Y es el mismo set — no hay split real, val ≈ train.

**Fix correcto:**
```python
all_files = sorted(Path(cache_dir).glob('*.npz'))
rng = np.random.default_rng(seed)
shuffled = list(all_files)
rng.shuffle(shuffled)
split = int(len(shuffled) * 0.9)
train_files, val_files = shuffled[:split], shuffled[split:]

train_chunks = load_chunks_from_files(train_files, mirror_prob=0.5, rng=rng)
val_chunks   = load_chunks_from_files(val_files,   mirror_prob=0.0, rng=rng)
```

**Crítico:** sin un split real, `val_acc` está sobrestimado (overfitting invisible). El run anterior reportó 80.4% — ese número es ruido si train==val.

---

### OPT 5 — Modelo más pequeño para el smoke test

Para validar el pipeline, usa un modelo de juguete (10× menos parámetros, 10× más rápido):

```python
# Smoke test config
model = LimbSequenceTransformer(
    input_dim=input_dim, d_model=64, n_heads=4, n_layers=2, ffn_dim=128
)
```

Solo para confirmar que loss baja y que se guarda. Después escala al modelo real.

---

### OPT 6 — Logs por step (ubicar crashes)

Añade tras cada batch:
```python
if n_batches % 50 == 0:
    logger.info(f'  step {n_batches}, loss={float(loss):.4f}')
```

Y al inicio de cada epoch:
```python
logger.info(f'=== Epoch {epoch+1}/{epochs} starting ===')
```

Si vuelve a crashear silenciosamente, sabes en qué step.

---

### OPT 7 — bf16 (opcional, riesgo medio)

MLX soporta `mx.bfloat16`. **No** lo uses en el primer run estable — primero confirma que fp32 funciona. Una vez tengas baseline:

```python
model.set_dtype(mx.bfloat16)
```

**Ahorro estimado:** 1.5–2× en memoria y velocidad. Pero puede degradar acc 0.5–1pp si el loss es sensible.

---

## 3. Plan de ejecución paso a paso (ESTE es el camino)

### Paso 0 — Pre-flight

```bash
cd /Users/rodrigo/dev/piu/piu-annotate_to_label_piu
ls artifacts/manual-chartstructs/visss-120524-eaware/ | wc -l   # ~4261
python3 -c "import mlx.core as mx; print(mx.__version__)"        # 0.29.3
df -h .                                                          # ≥ 5 GB libres
```

### Paso 1 — Aplicar los 4 FIXES + las 6 OPTS

Aplica las ediciones en orden:
1. `train_mlx.py`: borra `_flatten_params`, importa `LimbSequenceTransformer` de `mlx_architecture`, borra el duplicado, fix save (FIX 1), fix mirror (FIX 2), añade `load_all_chunks_cached`, bucketing (OPT 1), `mx.compile` (OPT 2), eval cada 8 steps (OPT 3), split real (OPT 4), logs por step (OPT 6).
2. `mlx_architecture.py`: fix attention mask (FIX 3).
3. Crea `cli/limbuse/cache_chunks.py`.

> Antes de cualquier run, corre los tests existentes:
> ```bash
> python3 -m pytest tests/test_mirror.py -v
> ```

### Paso 2 — Pre-cachear features (UNA VEZ, ~5 min)

```bash
python3 cli/limbuse/cache_chunks.py \
  --folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --out_dir artifacts/cache/mlx-singles/ --sd singles \
  2>&1 | tee cache_singles.log
```

**Verificación:** `ls artifacts/cache/mlx-singles/ | wc -l` ≈ 2100 (solo singles).

### Paso 3 — SMOKE TEST (3 min, NO opcional)

Usa modelo pequeño + 100 charts + 2 epochs. Si esto no funciona, no escales.

```bash
/usr/bin/time -l python3 cli/limbuse/train_mlx.py \
  --singles_or_doubles singles \
  --manual_chart_struct_folder artifacts/cache/mlx-singles/ \
  --out_dir artifacts/models/visss-mlx-smoke/ \
  --epochs 2 --batch_size 16 --lr 3e-4 --seed 0 \
  --limit_charts 100 \
  2>&1 | tee smoke_test.log
```

**Criterios de éxito (TODOS deben cumplirse):**
- ✅ El proceso termina sin crash.
- ✅ `train_loss` baja entre epoch 1 y 2.
- ✅ Se crea `artifacts/models/visss-mlx-smoke/singles-arrows_to_limb-mlx-best.safetensors`.
- ✅ `mx.load('...best.safetensors')` carga sin error (verifica con el comando de FIX 1).
- ✅ `maximum resident set size` (en el log de `time -l`) < 4 GB.

**Si falla:** lee el log, fix el bug específico, repite. **No avances.**

### Paso 4 — Singles entero (~12–18 min con OPTs)

```bash
/usr/bin/time -l python3 cli/limbuse/train_mlx.py \
  --singles_or_doubles singles \
  --manual_chart_struct_folder artifacts/cache/mlx-singles/ \
  --out_dir artifacts/models/visss-mlx/ \
  --epochs 20 --batch_size 32 --lr 3e-4 --seed 0 \
  2>&1 | tee out_mlx_singles_v2.log
```

Notar `--batch_size 32` (subido de 16 — con bucketing y `mx.compile` cabe).

**Watchdog en otra terminal:**
```bash
while true; do
  ps -p $(pgrep -f train_mlx.py) -o pid,rss,vsz,etime 2>/dev/null || break
  sleep 30
done
```

**Criterios:**
- val_acc ≥ 85% al epoch 5
- val_acc ≥ 90% al epoch 20
- RSS estable (no crece monotonamente)

### Paso 5 — Doubles (~25–35 min)

```bash
python3 cli/limbuse/cache_chunks.py \
  --folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --out_dir artifacts/cache/mlx-doubles/ --sd doubles

/usr/bin/time -l python3 cli/limbuse/train_mlx.py \
  --singles_or_doubles doubles \
  --manual_chart_struct_folder artifacts/cache/mlx-doubles/ \
  --out_dir artifacts/models/visss-mlx/ \
  --epochs 20 --batch_size 16 --lr 3e-4 --seed 0 \
  2>&1 | tee out_mlx_doubles_v2.log
```

(Doubles tiene 21 dims y secuencias más largas — mantén batch_size 16.)

### Paso 6 — Cargar y validar el modelo

Antes de borrar el LightGBM, confirma que el MLX se puede cargar e inferir:

```python
import mlx.core as mx
from mlx.utils import tree_unflatten
from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer

flat = mx.load('artifacts/models/visss-mlx/singles-arrows_to_limb-mlx-best.safetensors')
model = LimbSequenceTransformer(input_dim=18, d_model=128, n_heads=8, n_layers=4, ffn_dim=512)
model.update(tree_unflatten(list(flat.items())))
mx.eval(model.parameters())
print('Model loaded OK, n_params:', sum(v.size for v in flat.values()))
```

### Paso 7 — Benchmark (compara contra LGBM)

```bash
# Compara apples-to-apples con el benchmark existente
python3 scripts/benchmark_annotations.py --show_worst 30 --top_diff 30 --plot
```

Actualiza `docs/progress.md` con los números reales (no los proyectados).

---

## 4. Tabla de tiempos esperados (con todas las OPTs aplicadas)

| Fase | Antes (run fallido) | Después (esta guía) |
|---|---|---|
| Featurización (cache miss) | ~5 min/run × N runs | **~5 min total**, una vez |
| Featurización (cache hit) | n/a | **~10 s** |
| Smoke test (100 charts, 2 epochs) | n/a | **~2 min** |
| Singles training (4261 charts, 20 epochs) | ~33 min (sin save válido) | **~12–18 min** |
| Doubles training (20 epochs) | n/a (jamás llegó) | **~25–35 min** |
| Save/load roundtrip | ❌ no funciona | ✅ verificable |

**Speedup total agregado (ciclo iterar–entrenar–verificar):** **3–5×.**

---

## 5. Checklist final antes de lanzar

- [ ] `mx.__version__ == '0.29.3'`
- [ ] FIX 1 aplicado (mx.save_safetensors + tree_flatten)
- [ ] FIX 2 aplicado (mirror a nivel de chart)
- [ ] FIX 3 aplicado (attn_mask shape `(B, 1, L)`)
- [ ] FIX 4 aplicado (cache + lazy load)
- [ ] OPT 1 aplicado (length bucketing)
- [ ] OPT 2 aplicado (`mx.compile`)
- [ ] OPT 3 aplicado (eval cada 8 steps)
- [ ] OPT 4 aplicado (train/val split real)
- [ ] OPT 6 aplicado (logs por step + epoch)
- [ ] Tests de mirror pasan (`pytest tests/test_mirror.py`)
- [ ] Cache generado para singles
- [ ] Smoke test pasó los 5 criterios
- [ ] Modelo del smoke test cargable con `mx.load`
- [ ] Watchdog corriendo en segunda terminal antes del run completo
- [ ] LightGBM models respaldados (`artifacts/models/visss-backup-*/`)

---

## 6. Troubleshooting express

| Síntoma | Causa más probable | Acción |
|---|---|---|
| Crash silencioso < 1 min | OOM por chunks gigantes | Verifica que `load_all_chunks_cached` se está usando, no el viejo |
| Crash en epoch 1 step 0 | attn_mask shape | Confirma FIX 3 |
| `mx.load` falla con "invalid format" | Save format viejo | Confirma FIX 1, regenera modelo |
| val_acc no sube de 60% | Mirror corrompido o val=train | Confirma FIX 2 + OPT 4 |
| RSS crece linealmente | Cache no se está usando, re-featurizando | Mira los logs — debe decir "Loading cached" |
| `train_loss = nan` | LR muy alto o gradient explosion | Baja `--lr 1e-4` o reactiva grad clip a 0.5 |
| Smoke test pasa, full crashea | `batch_size 32` muy grande para singles full | Vuelve a 16 |

---

## 7. Anexo: por qué este plan no rompe nada existente

| Pieza | Estado |
|---|---|
| LightGBM training (`train_lgbm.py`) | **Intacto** — no se toca |
| LightGBM models (`artifacts/models/visss/`) | **Intacto** — sigue en producción |
| `process_db_matches.py` | **Intacto** — sigue usando LGBM |
| `benchmark_annotations.py` | **Intacto** — funciona contra cualquier output |
| MLX models | **Aislados** en `artifacts/models/visss-mlx/` |
| Cache de features | **Nuevo** en `artifacts/cache/`, no colisiona |
| `mirror.py` | **Intacto** — solo se usa diferente en train |
| `mlx_architecture.py` | **1 fix** (attn mask), arquitectura igual |

Si MLX falla de nuevo, **simplemente borra `artifacts/models/visss-mlx/` y `artifacts/cache/`**. El pipeline LGBM sigue funcionando exactamente igual.

---

## 8. Resumen ejecutivo (copy-paste)

```bash
# 0) Aplicar fixes en train_mlx.py + mlx_architecture.py (ver §1)
# 1) Crear cli/limbuse/cache_chunks.py (ver FIX 4)
# 2) Cachear (UNA VEZ)
python3 cli/limbuse/cache_chunks.py --folder artifacts/manual-chartstructs/visss-120524-eaware/ --out_dir artifacts/cache/mlx-singles/ --sd singles
python3 cli/limbuse/cache_chunks.py --folder artifacts/manual-chartstructs/visss-120524-eaware/ --out_dir artifacts/cache/mlx-doubles/ --sd doubles

# 3) Smoke test
/usr/bin/time -l python3 cli/limbuse/train_mlx.py --singles_or_doubles singles --manual_chart_struct_folder artifacts/cache/mlx-singles/ --out_dir artifacts/models/visss-mlx-smoke/ --epochs 2 --batch_size 16 --limit_charts 100 2>&1 | tee smoke_test.log

# 4) Verificar carga
python3 -c "import mlx.core as mx; d = mx.load('artifacts/models/visss-mlx-smoke/singles-arrows_to_limb-mlx-best.safetensors'); print(f'OK {len(d)} tensors')"

# 5) Singles full
/usr/bin/time -l python3 cli/limbuse/train_mlx.py --singles_or_doubles singles --manual_chart_struct_folder artifacts/cache/mlx-singles/ --out_dir artifacts/models/visss-mlx/ --epochs 20 --batch_size 32 --lr 3e-4 2>&1 | tee out_mlx_singles_v2.log

# 6) Doubles full
/usr/bin/time -l python3 cli/limbuse/train_mlx.py --singles_or_doubles doubles --manual_chart_struct_folder artifacts/cache/mlx-doubles/ --out_dir artifacts/models/visss-mlx/ --epochs 20 --batch_size 16 --lr 3e-4 2>&1 | tee out_mlx_doubles_v2.log

# 7) Benchmark
python3 scripts/benchmark_annotations.py --show_worst 30 --top_diff 30 --plot
```
