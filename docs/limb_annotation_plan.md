# Limb Annotation: Estado actual, charts sin cobertura y plan ML

## TL;DR

| Grupo | Cantidad | Calidad |
|-------|----------|---------|
| Charts con vis-ss (ground truth manual) | 3 703 | ✅ correctos |
| Charts SIN vis-ss (regla + naturalness fix) | 660 | ~82% tap accuracy |

Los 3 703 ya están bien — tienen anotaciones manuales del vis-ss exportadas directamente.
Los 660 de abajo usan el algoritmo rule-based mejorado. Estos son candidatos para ML.

---

## ¿Qué mejoró el algoritmo rule-based?

Pipeline actual (`predict_limbs_pattern_only` en `cli/ingest/process_db_matches.py`):

1. **PatternReasoner** — detecta runs de pasos únicos (alternating/same). Solo actúa en filas con `num_downpress == 1`.
2. **Alternating fill** — rellena posiciones abstained con L/R alternante.
3. **`_fix_multihits_by_naturalness`** — para cada fila con 2+ notas simultáneas, elige el combo de pies válido con mayor score de posición natural (paneles izquierdos → pie izquierdo, derechos → pie derecho). Tiebreak: menor distancia Hamming al combo actual.

**Mejora medida (benchmark vs vis-ss, 3 704 charts):**

| Categoría | Antes | Después |
|-----------|-------|---------|
| Triples | 50.1% | 83.0% |
| Jacks/repeated | 58.8% | 70.9% |
| Overall tap | 77.0% | 82.0% |

**Límite del rule-based:** Jacks vs footswitches requieren contexto temporal (qué pie usé antes, cuánto tiempo pasó). Eso necesita ML.

---

## Charts sin vis-ss (660 charts / 237 canciones)

Estos usan solo el algoritmo. Sin ground truth para validar.

### Canciones con más charts sin cobertura

| Canción | Charts faltantes |
|---------|-----------------|
| The Apocalypse | D13, D16, D18, D20, D23, S4, S7, S11, S14, S16, S18, S20 |
| SWEET WONDERLAND | D14, D18, D22, S3, S5, S8, S12, S15, S17, S20 |
| Ultimate Eyes | D13, D21, D24, S4, S7, S11, S14, S16, S19, S22 |
| Alice in Misanthrope | D13, D18, D21, D24, S12, S15, S17, S20, S22 |
| ALiVE | D13, D17, D21, D24, S5, S12, S15, S18, S22 |
| About The Universe | D18, D21, D24, S7, S12, S16, S18, S21 |
| Halloween Party ~Multiverse~ | D13, D21, D23, S8, S12, S16, S18, S20 |
| Becouse of You | D12, D17, D20, D22, S11, S14, S16, S18, S21 |
| Deca Dance | D17, D20, D23, S9, S13, S16, S18, S21 |
| Super Akuma Emperor | D18, D24, D26, S15, S18, S20, S22, S24 |

### Lista completa (por canción)

```
1948: D27, S18, S21, S24, S26
1949: D28
1950: D25, S23
4NT: D14, D20, D24, S9, S12, S16, S20, S22
8 6: S20
About The Universe: D18, D21, D24, S7, S12, S16, S18, S21
After LIKE: D20, D23
Alice in Misanthrope: D13, D18, D21, D24, S12, S15, S17, S20, S22
ALiVE: D13, D17, D21, D24, S5, S12, S15, S18, S22
All I Want For X-mas: D16, D18, S15, S17
Allegro Con Fuoco: S5
Allegro Più Mosso: S17
Altale: S6
An Interesting View: S13
Another Truth: S6
Aragami: D26
Arcana Force: D10, D17, D20, S4, S9
Arch of Darkness: D18, S5
Avalanquiem: S23
Baroque Virus - FULL SONG -: D23
BATTLE NO.1: D24
Beat of The War: D20, D24
Beat of The War 2: S17
Becouse of You: D12, D17, D20, D22, S11, S14, S16, S18, S21
Bee: S17
Beethoven Influenza: D20
Beethoven Virus: S16
Binary star: D15, D19, D23, S5, S13, S17, S20
Black Dragon: D23
Blaze Emotion: S3
Brown Sky: D24
Bullfighter's Song: S11
Burn Out: D20, D23, S11, S15, S17, S20
Can-can ~Orpheus in The Party Mix~: D12, D15, D18, D23, S16, S21
Can-can ~Orpheus in The Party Mix~ - SHORT CUT -: D17, D19, D21, D23
Canon D: D23
Caprice of Otada: D19, S17, S19
CARMEN BUS: D13, D20, S12, S18
CHICKEN WING: S20
Chimera: D26, S9, S23
Chobit Flavor: S12, S19, S22
Chopsticks Challenge: S17
Club Night: D23
Come to Me: S4, S6, S11
Conflict: D18, D25, D26
Cross Time: D22
Csikos Post: S4
D: D20
Dance with me: S4
Danger & Danger: D10, D14, D21, S16, S19, S21
Darkside Of The Mind: S19
Dead End: D22, D26, D28, S18, S21, S23, S25
Death Moon - SHORT CUT -: D23
Deca Dance: D17, D20, D23, S9, S13, S16, S18, S21
Dement ~After Legend~: D20, D26, S23
Demon of Laplace: D15, D20, D23, S6, S10, S17, S20
Desaparecer: D25
Destr0yer: D20, D24
Dignity: D17, D24, S21, S22
DJ Otada: D20
Dr. M: D18, D20, S3, S16
Dream To Nightmare: S21
Dual Racing <RED vs BLUE>: D13, D19, D21, S16, S18
DUEL - SHORT CUT -: D23, S16, S19, S21
Earendel: S20
Energy Synergy Matrix: D22
ERRORCODE: 0: D27, S23, S25
ESCAPE: D19, D26
ESP: S22
Eternal Universe: D19, D23, D25, S4, S7, S12, S16, S20, S23
Etude Op 10-4: D25, S22
Euphorianic: D21, D23
Extravaganza: D18, S4, S19
Faster Z: D21, S4
Final Audition: D19, S18
Final Audition 2: S18
Final Audition 3: D18, S16, S17
Final Audition Ep. 1: D15, S17, S21
Final Audition Ep. 2-1: S8, S19
Final Audition Ep. 2-2: D19, S17
First Love: D15
Flavor Step!: D22
Forgotten Vampire: D20, S8
Gargoyle - FULL SONG -: S21, S23
Ghroth - SHORT CUT -: D20, D24, S14, S18, S22
Giselle: D24
Glimmer Gleam: D14, D19, D23, S8, S13, S18, S21
GOODBOUNCE: S20
GOODTEK: D16, S8
Halloween Party ~Multiverse~: D13, D21, D23, S8, S12, S16, S18, S20
Halloween Party ~Multiverse~ - SHORT CUT -: D21, S14, S19, S21
Hardkore of the North: S22
Heart Attack: D21
Heliosphere: D20, D23, D25, S14, S18, S21, S23
HELIX: D23
Hi Bi: D20, S16, S19
Higgledy Piggledy: S2
HTTP: D24
Human Extinction (PIU Edit.): D20, D23, D25, S12, S18, S20, S22, S24
HUSH: D15, S5, S15
HUSH - FULL SONG -: S15
Hymn of Golden Glory - SHORT CUT -: D17, D20, D22, D24, S15, S18, S20, S23
Hypnosis(SynthWulf Mix): S11, S13, S20
Ignis Fatuus(DM Ashura Mix): D17, D18, S12, S16, S19
Imaginarized City: S21
Imprinting: D21
J Bong: S15
Jam O Beat: D15, S16
Jupin: D25, S23
Jupin - SHORT CUT -: D19, D24, S12, S13, S14, S21, S22, S23
K.O.A : Alice In Wonderworld: S3
K.O.A : Alice In Wonderworld - SHORT CUT -: S16
Katkoi: D24, S22
Kill Them!: D19, S15, S18
Kitty Cat: S1
Kokugen Kairou Labyrinth: D26, S20, S23
KUGUTSU: D27
La Cinquantaine: D24, S22
Ladybug: S3
Le Nozze di Figaro ~Celebrazione Remix~: D21
Leakage Voltage: D23
Little Munchkin: D17
Love is a Danger Zone: D17, S4, S11, S17
Love is a Danger Zone pt. 2: D18, D24
Love Is A Danger Zone(Cranky Mix): D16, D23, S7, S14, S20
Magical Vacation: D21
Mental Rider: D22, S16
Meteorize: D19
MilK: S15
Moment Day: D23
Monolith: D14
Move That Body!: D18, S3, S12, S17, S20
Move That Body! - FULL SONG -: D19, S18
Move That Body! - SHORT CUT -: D18
Mr. Larpus: S3
msgoon RMX pt.6: D21
Murdoch vs Otada: D24, S19
Murdoch vs Otada - SHORT CUT -: D20, D24, S15, S18, S21
My Dreams: S19
My way: S15
Napalm: D16
Neo Catharsis: D21
Neo Catharsis - SHORT CUT -: D23, D25, S19, S21, S23
Night Duty: S6, S17
Oh! Rosa: S9
PANDORA: D6, D13, D18, S2, S4, S7, S11
Papasito (feat. KuTiNA): D11, D14, D21, S7, S13, S15
Papasito (feat. KuTiNA) - FULL SONG -: D14, D20, S12
Paradoxx: D25, D28, S21, S26
Paradoxx - SHORT CUT -: D26, S21
Passacaglia: D14, D25, S8, S17
Passing Rider: S12, S16, S19
Phantom: S18, S19
Phoenix Opening - SHORT CUT -: D16, D20, S12, S16, S18
Pirate: D21
PRiMA MATERiA: S21
PRiMA MATERiA - SHORT CUT -: D22, D24, S18, S21, S23
Pump Jump: D18
Pump Me Amadeus: D24
Pumping Jumping: S4
Pumping Up: S5, S10
Pumptris Quattro: S18, S20
Radetzky Can Can: S12
Rage of Fire: D22, S18, S20
Ragnarok: S21
Reality: S4
Red Swan: D20
Repeatorment Remix: D22
Repentance: S10, S16, S19, S22
Re：End of a Dream: D18, D25, S15, S19, S21
Rolling Christmas: S3, S17
Rush-Hour: D17, D22, D24, S13, S16, S20, S22
Sarabande: S19
See: S16, S22
Set me up: S4
Shub Sothoth: D27
Silver Beat feat. ChisaUezono: D22
Slapstick Parfait: D23
Solfeggietto: S21
Solitary: S2, S17, S18
Solitary 1.5: D18
Solitary 2: D20, S9, S18
Solve My Hurt: D15, D19, D23, S10, S14, S18, S21
Solve My Hurt - SHORT CUT -: D26, S20, S23
Sorceress Elise: D13
Spooky Macaron: D16, D18, D22, D25, S14, S17, S20, S23
STAGER: D20
Stardream -Eurobeat Remix- - SHORT CUT -: D20, D22, S12, S16, S18, S21
STEP: S7
Street show down: D19, S16
Sudden Appearance Image: D23, S21
Sugar Plum: S19
Super Akuma Emperor: D18, D24, D26, S15, S18, S20, S22, S24
Super Fantasy: D22
SWEET WONDERLAND: D14, D18, D22, S3, S5, S8, S12, S15, S17, S20
Take Out: D13
Teddy Bear: D18
Teddy Bear - FULL SONG -: D22, S12, S20
Tek -Club Copenhagen-: S5
Tepris: D20, S17
That Kitty (PIU Edit.): D15, D22, D24, S13, S17, S20, S23
The Apocalypse: D13, D16, D18, D20, D23, S4, S7, S11, S14, S16, S18, S20
The People didn't know: S16
The Reverie: D22
THE REVOLUTION: D19, S7, S12, S17, S19
Till the end of time: S3
Toccata: D18, S17
TOMBOY: D20
Top City: S13
Tribe Attacker: S4
TRICKL4SH 220: D23
Turkey March: S17
U GOT 2 KNOW: S2, S17
Ugly duck Toccata: D18, S17
Ultimate Eyes: D13, D21, D24, S4, S7, S11, S14, S16, S19, S22
Ultimatum: S17, S21, S23, S25
Unique: D22
Up & Up (Produced by AWAL): D10, S3, S17
Uranium: S22
Utopia: D22, S4, S7, S10, S20
VANISH: D24
Vanish 2 - Roar of the invisible dragon: D18, D24, S12, S17, S20, S22
VECTOR: D24, S22
Versailles: D23
What Are You Doin?: D23
Wicked Legend: D21
Will-O-The-Wisp: D20, S19
Winter: D21, S7, S16, S20
Witch Doctor: D23, S11, S19, S21
Witch Doctor #1: S2, S5, S8, S9, S17, S18, S19
With My Lover: D14, S12
X-Tream: D15, S5
XTREE: S2
Xuxa: D12, D17, S3, S5, S14
Yog-Sothoth: D24
You again my love: D15, S1, S6, S14
†DOOF†SENC†: D16, D19, D23, D25, S15, S18, S21, S23
```

---

## Plan ML: entrenar el modelo aprovechando la infra existente

> **Hallazgo clave.** El repo ya tiene un pipeline ML completo (autor original
> maxwshen) en `piu_annotate/ml/` y `cli/limbuse/`. NO hay que construir el
> entrenamiento desde cero. Falta solo: (a) generar el dataset desde vis-ss en
> el formato que espera el pipeline (ChartStruct CSV con `Limb annotation`
> poblada) y (b) cablear el `ModelSuite` entrenado en `process_db_matches.py`.

### Por qué el rule-based falla en jacks

Jack (golpe repetido en mismo panel) admite dos ejecuciones:
- **Jack real**: mismo pie dos veces (L-L o R-R) — posible si hay tiempo
- **Footswitch**: pie alternado en mismo panel (L-R o R-L) — obligatorio a BPM alto

Rule-based no tiene contexto temporal suficiente. El ML sí.

### Infra ML existente (no reescribir)

| Archivo | Rol |
|---------|-----|
| `piu_annotate/ml/datapoints.py` | `ArrowDataPoint` (features por flecha) y `LimbLabel` (target 0=L / 1=R). Ya define features categóricas: `arrow_pos`, `arrow_symbol`, `has_active_hold`, `time_since_last_same_arrow_use`, `time_since_prev_downpress`, `num_downpress_in_line`, `line_is_bracketable`, `line_repeats_previous_downpress_line`, `line_repeats_next_downpress_line`, y la línea entera `cat.line_posN`. |
| `piu_annotate/ml/featurizers.py` | `ChartStructFeaturizer` arma matrices con ventana de contexto (`ft.context_length=20`, `ft.prev_limb_context_len=8`). Expone `featurize_arrows_with_context` (no usa pies previos) y `featurize_arrowlimbs_with_context` (sí los usa). |
| `piu_annotate/ml/models.py` | `LGBModel` (wrapper LightGBM) y `ModelSuite` que carga 4 modelos por cada `singles`/`doubles`. |
| `piu_annotate/ml/tactics.py` | `Tactician` — combina los 4 modelos en una predicción iterativa con scoring (`enforce_arrow_after_hold_release`, búsqueda de mejor secuencia). |
| `piu_annotate/ml/predictor.py` | `predict(cs, model_suite)` — orquesta `PatternReasoner` + `Tactician`. Punto de entrada para inferencia. |
| `cli/limbuse/train_lgbm.py` | Entrena el `ModelSuite` (4 modelos) sobre un folder de CSVs ChartStruct con columna `Limb annotation`. |
| `cli/limbuse/predict_limbs.py` | Inferencia masiva con `ModelSuite`. |
| `cli/limbuse/eval_models.py` | Evaluación de modelos. |

**El `ModelSuite` entrena 4 modelos por separación singles/doubles:**

1. `arrows_to_limb` — predicción base de pie (sin usar pies previos)
2. `arrowlimbs_to_limb` — refinamiento usando pies ya asignados como features
3. `arrows_to_matchnext` — ¿el pie de esta nota coincide con la siguiente?
4. `arrows_to_matchprev` — ¿coincide con la anterior?

El `Tactician` los combina iterativamente. Esto es lo que da accuracy alta en jacks/footswitches: los modelos `matchnext`/`matchprev` capturan transiciones que el modelo base solo no resuelve.

### Datos de entrenamiento

**Fuente:** 3 703 charts vis-ss en `/home/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524/`.

**Formato vis-ss** (cada `.json` es `[taps, holds, metadata]`):
- tap: `[panel, time, "l"|"r"]`
- hold: `[panel, start_time, end_time, "l"|"r"]`
- metadata: dict con `shortname`, `STEPSTYPE`, `Manual limb annotation` (bool), etc.

**Formato esperado por `train_lgbm.py`:** ChartStruct CSV con columnas
`Beat`, `Time`, `Line`, `Line with active holds`, `Limb annotation`. La
`Limb annotation` es string concatenado (`l`/`r`/`e`/`h`/`?`) cuyos
caracteres se alinean con los símbolos no-cero de `Line with active holds`
en el mismo orden de panel.

**Filtro recomendado:** usar solo charts con `metadata["Manual limb annotation"] == true` para garantizar ground truth humano. Si quedan pocos, ampliar con auto-anotados.

### Pipeline de entrenamiento (concreto)

**Paso 1 — Conversión vis-ss → ChartStruct CSV con limbs**

Nuevo: `scripts/visss_to_chartstruct.py`. Por cada JSON vis-ss:

1. Determinar singles/doubles desde `metadata["STEPSTYPE"]` (`pump-single` → 5 paneles, `pump-double` → 10).
2. Agrupar taps por `time` exacto (multihits) y derivar el conjunto de paneles activos por línea.
3. Recorrer holds para marcar `2` (start), `3` (end), `4` (active hold) en `Line with active holds` siguiendo la convención de `ChartStruct` (`piu_annotate/formats/chart.py:50`).
4. Para cada línea, construir `Limb annotation` ordenando los símbolos no-cero por posición de panel ascendente y mapeando `l`/`r` desde el JSON. Para holds activos sin downpress nuevo, propagar el pie del start.
5. Calcular `Beat` desde `Time` usando los BPMs/offset del SSC original — o caer en aproximación constante si no es accesible (los modelos no usan `Beat` directamente, lo usan vía `time_since_*` que sí necesita `Time`).
6. Escribir `<shortname>.csv` (el nombre debe codificar singles/doubles porque `train_lgbm.py:guess_singles_or_doubles_from_filename` lo lee del filename: `..._S16_ARCADE.csv` / `..._D20_ARCADE.csv`).

Output: `artifacts/manual-chartstructs/visss-120524/*.csv` (~3 703 archivos).

**Paso 2 — Entrenar el `ModelSuite` (singles y doubles por separado)**

```bash
python cli/limbuse/train_lgbm.py \
    --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524/ \
    --singles_or_doubles singles \
    --out_dir artifacts/models/visss/ \
    --model lightgbm

python cli/limbuse/train_lgbm.py \
    --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524/ \
    --singles_or_doubles doubles \
    --out_dir artifacts/models/visss/ \
    --model lightgbm
```

Cada corrida produce 4 archivos `.txt` (modelos LightGBM serializados):
`{singles|doubles}-arrows_to_limb.txt`,
`{singles|doubles}-arrowlimbs_to_limb.txt`,
`{singles|doubles}-arrows_to_matchnext.txt`,
`{singles|doubles}-arrows_to_matchprev.txt`.

Fix antes de correr: `train_lgbm.py:52` y `:140` apuntan al home de
`maxwshen`. Cambiar a paths locales o pasarlos via `args` con `hackerargs`.

**Paso 3 — Validación cruzada por canción**

`train_lgbm.py` usa `train_test_split` por nota (`random_state=0`) — sesgo
optimista (notas de la misma canción quedan en train y test). Mejorar:
agrupar splits por `shortname` o por canción (usar `GroupKFold` de sklearn).
Idealmente reservar 10% de canciones como hold-out estricto y reportar
accuracy por categoría (single / jack / triple / hold) reusando la
clasificación de `scripts/benchmark_annotations.py`.

**Paso 4 — Cablear inferencia en `process_db_matches.py`**

Reemplazar `predict_limbs_pattern_only(cs)` por:

```python
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict as ml_predict

# cargar UNA vez fuera del loop
suite_singles = ModelSuite('singles')
suite_doubles = ModelSuite('doubles')

def predict_limbs_ml(cs):
    suite = suite_singles if cs.singles_or_doubles() == 'singles' else suite_doubles
    ml_predict(cs, suite)
    # post-process: naturalness sigue siendo necesario para multihits
    pred_limbs = cs.get_limb_array()
    pred_coords = cs.get_prediction_coordinates()
    pred_limbs = _fix_multihits_by_naturalness(pred_limbs, pred_coords)
    cs.set_limb_array(pred_limbs)
```

Razón de mantener naturalness: el `Tactician` predice por flecha y puede
emitir combos inválidos en triples/brackets (p.ej. dos pies izquierdos en
paneles separados imposibles de pisar). El post-procesador actual ya
resuelve esto y no rompe nada cuando el modelo acierta.

Args necesarios: `model.dir`, `model.arrows_to_limb-singles`,
`model.arrowlimbs_to_limb-singles`, `model.arrows_to_matchnext-singles`,
`model.arrows_to_matchprev-singles` (ídem doubles). Ver `models.py:51-54`.

**Paso 5 — Regenerar los 660 sin vis-ss y sincronizar**

```bash
python cli/ingest/process_db_matches.py --only_missing_visss
python sync_to_piulatam.py  # ya filtra a esos 660
```

Validar que los 3 703 con vis-ss directo NO se toquen.

### Métricas objetivo

| Categoría | Actual rule-based | Objetivo ML | Cómo medir |
|-----------|------------------:|------------:|------------|
| Overall tap | 82% | >90% | `scripts/benchmark_annotations.py` (re-correr) |
| Triples | 83% | >90% | misma categoría en benchmark |
| Jacks/repeated | 71% | >85% | misma categoría |
| Hold limb | ~95% | >97% | añadir categoría hold al benchmark |

**Hold-out estricto:** entrenar dejando 10% de canciones fuera; reportar
accuracy en ese set para evitar engañarse con leakage por canción.

### Riesgos / decisiones abiertas

- **Calidad ground truth:** vis-ss marca `Manual limb annotation` por chart. Charts con `false` son auto-generados (no son verdad absoluta). Decidir si filtrar o usar todo.
- **Bracket post-process:** si entrenamos con verdad de brackets, el modelo aprende combos válidos solo y la naturalness puede sobrar. Medir antes de quitarla.
- **Singles vs doubles:** datasets desbalanceados. Si doubles tiene <500 charts, considerar augmentation (mirror del chart con paneles intercambiados).
- **`hackerargs`:** los scripts existentes usan `args.setdefault` con paths absolutos del autor original (`/home/maxwshen/...`). Reemplazar antes de ejecutar o pasar todos via CLI.
- **`Beat` aproximada:** si la conversión vis-ss→ChartStruct no recupera `Beat` exacta (porque vis-ss solo tiene `Time`), revisar que ningún feature dependa de `Beat`. `ChartStructFeaturizer` parece usar solo `Time` vía `annotate_time_since_downpress`, pero confirmar antes de entrenar.

---

## Archivos clave

**Pipeline actual de ingesta**
| Archivo | Rol |
|---------|-----|
| `cli/ingest/process_db_matches.py` | Pipeline principal: SSC → ChartStruct → limbs → JSON |
| `scripts/benchmark_annotations.py` | Benchmark rule-based vs vis-ss |
| `sync_to_piulatam.py` | Copia processed_db → piulatam (solo los 660 sin vis-ss) |
| `artifacts/processed_db/` | 4 372 charts procesados |
| `/home/rodrigo/dev/piu/piulatam/public/chart-jsons/` | Destino final del visualizador |
| `/home/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524/` | Ground truth vis-ss (4 124 charts) |

**Infra ML existente (reusar, no reescribir)**
| Archivo | Rol |
|---------|-----|
| `piu_annotate/ml/datapoints.py` | `ArrowDataPoint`, `LimbLabel` |
| `piu_annotate/ml/featurizers.py` | `ChartStructFeaturizer` con ventana de contexto |
| `piu_annotate/ml/models.py` | `LGBModel`, `ModelSuite` (4 modelos por singles/doubles) |
| `piu_annotate/ml/tactics.py` | `Tactician` — combina los 4 modelos |
| `piu_annotate/ml/predictor.py` | `predict(cs, suite)` — orquesta inferencia |
| `cli/limbuse/train_lgbm.py` | Entrena el `ModelSuite` desde folder de CSVs |
| `cli/limbuse/predict_limbs.py` | Inferencia masiva |
| `cli/limbuse/eval_models.py` | Evaluación |
| `piu_annotate/formats/chart.py` | `ChartStruct` (formato CSV target) |

---

## Tareas pendientes

**Entrenamiento del modelo**
- [ ] `scripts/visss_to_chartstruct.py` — convertir vis-ss JSON → CSV ChartStruct con `Limb annotation` poblada (filename codifica S/D)
- [ ] Ajustar paths hardcodeados de `cli/limbuse/train_lgbm.py` (`/home/maxwshen/...` → locales)
- [ ] Cambiar `train_test_split` por canción (`GroupKFold` por `shortname`) para evitar leakage
- [ ] Entrenar `ModelSuite` para `singles` y `doubles` apuntando a `artifacts/manual-chartstructs/visss-120524/`
- [ ] Extender `scripts/benchmark_annotations.py` con categoría hold y correr contra los modelos entrenados

**Integración**
- [ ] Reemplazar `predict_limbs_pattern_only` en `process_db_matches.py` por inferencia con `ModelSuite` + `Tactician` + naturalness post-process
- [ ] Cargar `ModelSuite` una sola vez fuera del loop (singles + doubles) para no pagar el pickle por chart

**Regeneración**
- [ ] Regenerar los 660 charts sin vis-ss con el modelo (no tocar los 3 703 con vis-ss directo)
- [ ] Sync solo esos 660 a piulatam
- [ ] Verificar accuracy spot-check de 5–10 charts (jacks de canciones difíciles: SWEET WONDERLAND S20, The Apocalypse S20)

**Opcional**
- [ ] Conseguir vis-ss para las 237 canciones faltantes si publican nueva versión
- [ ] Augmentation: espejo de doubles (intercambiar paneles 0↔9, 1↔8, etc.) para duplicar dataset doubles
