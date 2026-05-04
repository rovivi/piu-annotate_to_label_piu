# Guía paso a paso: re-entrenar con la regla de centro ambiguo en triples

Esta guía es a prueba de mensos. Cada paso incluye exactamente qué comando
correr, qué esperar, y cómo verificar que funcionó antes de pasar al siguiente.

**Hardware asumido:** Apple Silicon (M-series), 24 GB RAM.
**Tiempo total estimado:** 25–40 min (la mayor parte es el training de doubles).

---

## Pre-flight: contexto rápido

La regla nueva: en triples/quads que matcheen uno de estos patrones por pad
```
11100   01110   00111   10101   10110   01011
```
el panel CENTRAL (panel 2 en singles; paneles 2 y 7 en doubles) se etiqueta
como `e` (either-foot) **solo si**:
- (a) el centro se repite vs la fila anterior (jack en el centro), **o**
- (b) la fila anterior tenía un downpress en otra panel anotado `l`/`r`
      (ya hay orientación previa establecida).

Si ninguna de las dos se cumple, la anotación queda igual (no cambiamos charts
fáciles donde el inicio del triple sí marca orientación).

El helper vive en `piu_annotate/formats/notelines.py`:
- `multihit_ambiguous_panels(arrow_positions)` → set de panels ambiguos.
- `relabel_with_ambiguous_e(line, annot, prev_line, prev_annot)` → annot nueva.

---

## Paso 0 — Verifica el estado del repo

```bash
cd /Users/rodrigo/dev/piu/piu-annotate_to_label_piu
ls artifacts/manual-chartstructs/visss-120524/ | wc -l        # debe ser ~4261
ls artifacts/models/visss/ | grep -E '^(singles|doubles)-'    # 8 .txt files
```

**Esperado:** 4261 CSVs en el folder de training, 8 modelos entrenados (4
singles + 4 doubles).

Si ves 0 CSVs, regenera con
`python3 scripts/visss_to_chartstruct.py` antes de continuar.

---

## Paso 1 — Smoke test del helper de re-etiquetado

```bash
python3 -c "
from piu_annotate.formats import notelines
# triple 01110, prev tenía l en panel 0 -> debe cambiar centro a e
print('case A:', notelines.relabel_with_ambiguous_e('\`01110','llr','\`10000','l'))
# triple 01110, sin prev -> no cambia
print('case B:', notelines.relabel_with_ambiguous_e('\`01110','llr', None, None))
# 01011 (panel 2 NO está activo) -> no cambia
print('case C:', notelines.relabel_with_ambiguous_e('\`01011','lrr','\`10000','l'))
"
```

**Esperado exactamente:**
```
case A: ler
case B: llr
case C: lrr
```

Si ves esto → la regla funciona. Si no, **detente** y lee el error: usualmente
es un import o un cambio que no se guardó.

---

## Paso 2 — Generar el dataset re-etiquetado (visss-120524-eaware)

```bash
python3 scripts/relabel_visss_with_e_centers.py
```

**Lo que hace:** lee todos los CSV de `visss-120524`, aplica la regla fila por
fila respetando el contexto previo (importante: el `prev_annot` que se usa es
el ya re-etiquetado, no el original — así un jack en el centro se propaga
correctamente), y escribe los CSV nuevos en
`artifacts/manual-chartstructs/visss-120524-eaware/`.

**Tiempo:** ~30–60 s.

**Esperado al final del log:**
```
Done. 4261 files written -> .../visss-120524-eaware. <N> files modified, <M> rows relabeled.
```

`N` debería ser del orden de cientos a algunos miles (no 0, no 4261). Si N=0,
algo está mal con el helper o el dataset. Si N≈4261, la regla está aplicando
demasiado.

**Verificación rápida** (antes vs después):
```bash
python3 -c "
import pandas as pd, glob
old = sum(1 for f in glob.glob('artifacts/manual-chartstructs/visss-120524/*.csv')[:200]
          for a in pd.read_csv(f, dtype={'Limb annotation':str})['Limb annotation'].fillna('')
          if 'e' in a)
new = sum(1 for f in glob.glob('artifacts/manual-chartstructs/visss-120524-eaware/*.csv')[:200]
          for a in pd.read_csv(f, dtype={'Limb annotation':str})['Limb annotation'].fillna('')
          if 'e' in a)
print(f'rows con e (sample 200 csvs): old={old}  new={new}')
"
```
**Esperado:** `new > old`. Las `e` nuevas son las que añadiste.

---

## Paso 3 — Limpiar caché de datasets pre-featurizados

Si ya has corrido training antes, hay un cache que reusa features con hash
del nombre del folder. Vamos a forzar rebuild:

```bash
rm -rf cli/temp/dataset-storage/*.pkl.gz 2>/dev/null
echo "Cache limpio."
```

(El `2>/dev/null` evita un error si el folder no existe.)

---

## Paso 4 — Backup de modelos viejos

**No se puede deshacer un entrenamiento mal**. Mueve los modelos actuales a
una carpeta dated:

```bash
TIMESTAMP=$(date +%Y%m%d-%H%M)
mkdir -p artifacts/models/visss-backup-$TIMESTAMP
cp artifacts/models/visss/*.txt artifacts/models/visss-backup-$TIMESTAMP/
ls artifacts/models/visss-backup-$TIMESTAMP/   # debe listar los 8 .txt
```

Si quieres restaurar después:
`cp artifacts/models/visss-backup-$TIMESTAMP/*.txt artifacts/models/visss/`

---

## Paso 5 — Re-entrenar SINGLES

```bash
python3 cli/limbuse/train_lgbm.py \
  --manual_chart_struct_folder /Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524-eaware/ \
  --singles_or_doubles singles 2>&1 | tee out_singles_eaware.log
```

**Tiempo:** 5–10 min (4 sub-modelos: arrows_to_limb, arrowlimbs_to_limb,
matchnext, matchprev).

**Cosas que verificar en el log:**
- `Found <N> csvs in <M> directories` — debe ser ~4261.
- `Featurized <K> ChartStruct csvs` por cada sub-modelo — K ≈ 2100 (solo
  singles).
- Al final, una tabla de accuracy. Esperado: `arrows_to_limb val_acc ≥ 95%`,
  `arrowlimbs_to_limb val_acc ≥ 96%`. **Si val_acc < 90% algo se rompió** —
  abre el log y revisa.

Modelos guardados en `artifacts/models/visss/singles-*.txt`.

---

## Paso 6 — Re-entrenar DOUBLES

```bash
python3 cli/limbuse/train_lgbm.py \
  --manual_chart_struct_folder /Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524-eaware/ \
  --singles_or_doubles doubles 2>&1 | tee out_doubles_eaware.log
```

**Tiempo:** 15–25 min (doubles tiene más features y más ejemplos).

**Memoria:** doubles puede pegarle a 18–20 GB. Cierra apps pesadas (Chrome
con 30 pestañas, Slack, etc.) antes. Si el proceso muere por OOM, descomenta
las líneas 67–69 de `cli/limbuse/train_lgbm.py` que limitan a 460 CSVs.

Modelos guardados en `artifacts/models/visss/doubles-*.txt`.

---

## Paso 7 — Inferencia sobre el processed_db con los nuevos modelos

```bash
python3 cli/ingest/process_db_matches.py 2>&1 | tee out_processdb_eaware.log
```

**Tiempo:** 10–20 min (procesa ~4372 charts).

**Esperado:**
- Log final: `Processed N charts. Saved to artifacts/processed_db/`.
- No deberían aparecer warnings nuevos sobre `e`. Si aparece
  `Found impossible line with holds with no valid alternate`, anota el chart
  y revisa manualmente — la nueva regla puede estar cambiando algo en holds.

**Nota:** el post-process actual (`_fix_multihits_by_naturalness`) NO sabe de
`e`. Si el modelo predice `l`/`r` para esos centros, el post-process los
proyecta a un combo válido, igual que antes. Funciona sin cambios. Si
quieres que el output literal incluya `e` en el JSON visualizado, ese es un
cambio adicional que NO está en esta guía.

---

## Paso 8 — Benchmark

```bash
python3 scripts/benchmark_annotations.py --show_worst 30 --top_diff 30 --plot
```

Compara con los números actuales (de `docs/progress.md`):
| Métrica | Modelo actual | Esperado con la regla |
|---|---|---|
| Tap overall | 96.4% | 96–97% (sin cambio o leve mejora) |
| Jacks/repeated | 93.4% | 93–94% (igual) |
| Triple taps | 86.9% | **88–91%** (mejora real, era el target) |
| Hold accuracy | 97.0% | 97% (igual) |

**Importante:** el benchmark actual cuenta `e` como `l` (porque
`from_limb_annot` mapea `e→0`). Eso significa que cuando ground truth (de
vis-ss original sin `e`) dice `l` y el modelo predice cualquier cosa que
mapee a `l`, cuenta como correcto. La mejora viene de que el modelo ya no
recibe señal contradictoria entre charts donde el centro va `l` y donde va
`r` — antes le pedíamos clasificar lo inclasificable, ahora lo dejamos
abstener vía `e`.

Si quieres benchmark "estricto" (donde `e` matchee tanto `l` como `r` en
ref), eso requiere un cambio en `compare_chart()`; no está en esta guía.

---

## Paso 9 — Sync a piulatam (opcional)

```bash
python3 sync_to_piulatam.py
```

Esto copia los nuevos JSONs a `piulatam/public/chart-jsons/` para que la
UI los vea.

---

## Paso 10 — Commit (opcional)

```bash
cd /Users/rodrigo/dev/piu/piu-annotate_to_label_piu
git status
git add piu_annotate/formats/notelines.py \
        scripts/relabel_visss_with_e_centers.py \
        docs/retraining_guide.md \
        docs/progress.md
git diff --cached --stat
```

Si todo se ve bien:
```bash
git commit -m "Add ambiguous-center 'e' relabeling for triples"
```

---

## Troubleshooting

**Síntoma:** `relabel_visss_with_e_centers.py` falla con
`AttributeError: module 'piu_annotate.formats.notelines' has no attribute 'relabel_with_ambiguous_e'`.
→ El cambio en `notelines.py` no se guardó o estás en un venv viejo.
   Verifica: `python3 -c "from piu_annotate.formats import notelines; print(hasattr(notelines, 'relabel_with_ambiguous_e'))"` debe imprimir `True`.

**Síntoma:** Training de doubles muere con `Killed` (sin traceback).
→ OOM. Cierra apps, o limita csvs descomentando líneas 67–69 de `train_lgbm.py`.

**Síntoma:** `val_acc < 90%` en algún sub-modelo.
→ Probablemente borraste mal el cache y está reusando features viejos. Re-corre paso 3.

**Síntoma:** 0 rows relabeled en el paso 2.
→ El helper no detectó nada. Verifica que `notelines.py` tenga las funciones
   nuevas (paso 1) y re-corre.

**Síntoma:** Triples accuracy baja (no sube) después del re-entrenamiento.
→ Posiblemente N de rows relabeled fue muy bajo (< 50) y no es señal
   suficiente. Considera relajar la precondición (b) en
   `relabel_with_ambiguous_e` para que no requiera prev orientation, o
   relabel también triples sin prev. Esto es un experimento, no incluido
   por defecto.

---

## Resumen ejecutivo (para el yo del futuro)

```bash
# 1. Generar dataset nuevo con e-centers
python3 scripts/relabel_visss_with_e_centers.py

# 2. Limpiar cache + backup modelos
rm -rf cli/temp/dataset-storage/*.pkl.gz
mkdir -p artifacts/models/visss-backup-$(date +%Y%m%d)
cp artifacts/models/visss/*.txt artifacts/models/visss-backup-$(date +%Y%m%d)/

# 3. Re-entrenar
python3 cli/limbuse/train_lgbm.py \
  --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --singles_or_doubles singles
python3 cli/limbuse/train_lgbm.py \
  --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --singles_or_doubles doubles

# 4. Inferencia + benchmark
python3 cli/ingest/process_db_matches.py
python3 scripts/benchmark_annotations.py --show_worst 30 --top_diff 30 --plot
```

## Historial de Experimentos y Resultados

### [2026-05-04] - Optimización de Triple-Taps y Despliegue Masivo
**Configuración:** Relajación total de `e` centers (se aplica a todo centro en triple-tap) + Entrenamiento con dataset completo (mapping `e` -> `0`).

*   **Lo que se hizo:**
    *   Se eliminó la restricción de "contexto previo" para marcar `e`. Ahora cualquier centro en triple-tap es `e`.
    *   Se re-entrenó el suite LightGBM completo.
    *   Inferencia limpia de **4,418 charts** (637 canciones).
    *   Sincronización total con `piulatam`.

*   **Resultados (Benchmarks):**
    *   **Doubles:** **96.5%** (Mejora significativa en estabilidad).
    *   **Triple-taps:** **86.9%** (Recuperado tras el bajón del experimento anterior).
    *   **Singles:** **75.4%** (Identificado como el techo del modelo GBDT actual).

*   **Notas del Dev (Roadmap):**
    *   El modelo LightGBM ha llegado a su límite en Singles. No puede distinguir preferencias de bracket sin memoria de secuencia.
    *   **Siguiente gran paso:** Migración a **Transformers (MLX)** para procesar los charts como secuencias temporales completas.
    *   **Refinamiento Físico:** El Tactician necesita una penalización por rotación de cadera para evitar saltos imposibles reportados en los logs.

---

## MLX Transformer Migration — Attempt 2026-05-04

### Objetivo
Migrar de LightGBM (75.4% singles, 96.4% tap) a un `LimbSequenceTransformer` (MLX) que procese el chart completo como secuencia, con `e` (either foot) como clase separada (no forzada a `l`).

**Target:** 90%+ singles accuracy

### Lo que se implementó

| Componente | Archivo | Cambio |
|---|---|---|
| Mirror augmentation (horizontal flip) | `piu_annotate/formats/mirror.py` | `mirror_line`, `mirror_limb_annot`, `mirror_chartstruct` |
| Tests de mirror | `tests/test_mirror.py` | Round-trip tests |
| Transformer architecture | `piu_annotate/ml/mlx_architecture.py` | `LimbSequenceTransformer`: 4 capas, 8 heads, d_model=128, 3 classes (l/r/e) |
| Labels con `e` como clase 2 | `piu_annotate/ml/datapoints.py` | `LimbLabel.from_limb_annot`: `e→2` (antes `e→0`) |
| Loss: cross-entropy 3-class | `train_mlx.py` | log-softmax manual + take_along_axis (MLX 0.29.3 no tiene `mx.log_softmax`) |
| Model save/load | `train_mlx.py` | Params aplanados con dot-keys en `np.savez` |

### Crash Analysis — PID 92111 (2026-05-04)

**Síntoma:** Proceso corrió ~18 min, desapareció sin archivos de output, sin logs de error.

**Report completo:** `MLX_TRAINING_BUG_REPORT.md`

#### Bugs críticos encontrados

**BUG 1 (HIGH):** Guardar `.npz` con extensión `.safetensors`
```python
# train_mlx.py:299-310 — WRONG
save_path = ...f'{sd}-arrows_to_limb-mlx-best.safetensors'
np.savez(save_path, **flat_state)  # Produce .npz, no .safetensors
```
`mx.load(save_path)` fallaría. Fix: usar `mx.save` o `safetensors.numpy.save_file`.

**BUG 2 (HIGH):** Mirror augmentation mezclaba features originales con labels reflejadas
```python
# train_mlx.py:138-144
cx[:min_len] = mx2[:min_len]   # Sobrescribe features con mirror
cy[:min_len] = my[:min_len]    # Usa labels del chart mirrorado
# → desalineación feature/label si min_len < len(cx)
```
Fix: usar `cx_mirror` + `cy_mirror` consistentemente, o original completo.

**BUG 3 (MEDIUM):** Attention mask con forma incorrecta
```python
# train_mlx.py:89-92 — WRONG
attn_mask = mx.where(padding_mask[:, None, :, None], -1e9, 0.0)
# padding_mask es (B, L); indexing produce (B, 1, L, 1) que no broadcastea a (B, H, L, L)
```
Fix: `mx.where(padding_mask[:, None, :], -1e9, 0.0)` → `(B, 1, L)` que broadcastea correctamente.

**BUG 4 (MEDIUM):** Todos los chunks en RAM antes de entrenar
```python
# train_mlx.py:127 — all_chunks = []
for cs, label_col in ...:
    all_chunks.append((cx, cy))  # 4261 charts × 1-10+ chunks = gigabytes en RAM
```
Risk: OOM. El proceso llegó a 5.5GB VSIZE antes de morir.

**BUG 5 (LOW):** ~537 charts (~10%) skipados silenciosamente por "string index out of range" en loading.

#### Causa más probable del crash: OOM

El proceso creció a 5.5GB VSIZE y desapareció. En M-series con unified memory, macOS puede matar procesos silenciosamente cuando hay presión de memoria (SIGKILL sin handler, sin core dump).

Si fue OOM, probablemente ocurrió durante el primer `np.savez()` (escribir ~300MB de params) o en la última fase del epoch 1.

### Próximos pasos recomendados

1. **Fix BUG 1** (safetensor format) — necesario para que cualquier modelo se pueda cargar
2. **Fix BUG 2** (mirror augmentation) — training con datos corruptos
3. **Correr con monitoreo de memoria:**
   ```bash
   /usr/bin/time -l python cli/limbuse/train_mlx.py ... 2>&1 | tee training.log
   ```
4. **Añadir logging por epoch** para saber exactamente dónde muere
5. **Considerar data loading lazy** (generators en vez de materializar todo en RAM)

---

## Resumen: Guía rápida de re-entrenamiento (para el yo del futuro)

```bash
# 1. Generar dataset nuevo con e-centers (si cambió la regla)
python3 scripts/relabel_visss_with_e_centers.py

# 2. Limpiar cache + backup modelos
rm -rf cli/temp/dataset-storage/*.pkl.gz
mkdir -p artifacts/models/visss-backup-$(date +%Y%m%d)
cp artifacts/models/visss/*.txt artifacts/models/visss-backup-$(date +%Y%m%d)/

# 3A. Re-entrenar LightGBM (pipeline actual, estable)
python3 cli/limbuse/train_lgbm.py \
  --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --singles_or_doubles singles
python3 cli/limbuse/train_lgbm.py \
  --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --singles_or_doubles doubles

# 3B. Re-entrenar MLX Transformer (experimental, ver BUGs arriba)
#    AVISO: tiene BUGs críticos documentados arriba. Leer MLX_TRAINING_BUG_REPORT.md antes.
python3 cli/limbuse/train_mlx.py \
  --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --singles_or_doubles singles \
  --out_dir artifacts/models/visss-mlx/ \
  --epochs 20 --batch_size 16 --lr 3e-4

# 4. Inferencia + benchmark
python3 cli/ingest/process_db_matches.py
python3 scripts/benchmark_annotations.py --show_worst 30 --top_diff 30 --plot
```

