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

Listo.
