# piu-annotate

**Sistema de anotación de extremidades y predicción de dificultad para charts de Pump It Up**

Analiza charts del juego de ritmo Pump It Up (PIU), predice qué extremidad (pie izquierdo, pie derecho, mano) debe usar el jugador para cada flecha, segmenta charts en secciones de dificultad, y predice ratings de dificultad.

---

## Tabla de Contenidos

1. [Visión General del Sistema de Skills](#visión-general-del-sistema-de-skills)
2. [Conceptos Fundamentales](#conceptos-fundamentales)
3. [Sistema de Detección de Skills](#sistema-de-detección-de-skills)
   - [Run (Carrera)](#run-carrera)
   - [Drill](#drill)
   - [Jack](#jack)
   - [Footswitch](#footswitch)
   - [Bracket](#bracket)
   - [Staggered Bracket](#staggered-bracket)
   - [Doublestep](#doublestep)
   - [Twists](#twists)
4. [Sistema de Razonamiento de Patrones](#sistema-de-razonamiento-de-patrones)
   - [LimbReusePattern](#limbreusepattern)
   - [PatternReasoner](#patternreasoner)
5. [Problemas Conocidos y Mejoras Posibles](#problemas-conocidos-y-mejoras-posibles)
6. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
7. [Instalación](#instalación)

---

## Visión General del Sistema de Skills

El sistema de skills en `piu_annotate/segment/skills.py` detecta patrones técnicos en los charts de PIU. Cada skill se marca como una columna booleana en el DataFrame del ChartStruct (columnas con prefijo `__` como `__run`, `__footswitch`, etc.).

### Skills Detectados (25+ tipos)

| Skill | Descripción | Tipo |
|-------|-------------|------|
| `drill` | Notas alternadas rápido con mismo patrón rítmico | Básico |
| `run` | Secuencia de notas alternando pies | Básico |
| `anchor_run` | Run que empieza con pie "ancla" | Básico |
| `jack` | Misma nota, mismo pie, rápida | Básico |
| `footswitch` | Misma nota, pies alternados | Básico |
| `bracket` | Dos notas que caben en un pie | Posición |
| `staggered_bracket` | Bracket con notes desfasadas | Posición |
| `doublestep` | Dos notas diferentes para mismo pie | Posición |
| `hands` | Más de 2 notas que requieren manos | Posición |
| `jump` | Dos pies usados simultáneamente | Posición |
| `twist_90` | Twist de 90 grados | Twist |
| `twist_over90` | Twist de más de 90 grados | Twist |
| `twist_close` | Twist cercano | Twist |
| `twist_far` | Twist lejano | Twist |
| `side3_singles` | Notas en lado 3 en singles | Posición |
| `mid4_doubles` | Notas en medio de doubles | Posición |
| `mid6_doubles` | Notas en medio de doubles | Posición |
| `split` | Split (pies en extremos opuestos) | Posición |
| `stair5` | Escalera de 5 notas en singles | Patrón |
| `stair10` | Escalera de 10 notas en doubles | Patrón |
| `yog_walk` | Patrón de yog walk | Patrón |
| `cross_pad_transition` | Transición cruzando el pad | Patrón |
| `coop_pad_transition` | Transición co-op pad | Patrón |
| `hold_footswitch` | Holds con footswitch | Hold |
| `hold_footslide` | Hold con slide del pie | Hold |
| `bracket_run` | Run dentro de bracket | Compuesto |
| `bracket_drill` | Drill dentro de bracket | Compuesto |

---

## Conceptos Fundamentales

### ChartStruct y Líneas

El ChartStruct tiene una fila por "línea" (beat/timestamp). Cada línea tiene:

```
Line: `10000     → 5 caracteres para singles (01234 = vacio/flecha/inicio-hold/fin-hold)
Line with active holds: `10400  → incluye '4' para holds activos

Limb annotation: "lr"  → 'l'=pie izq, 'r'=pie der, para cada flecha en la línea
```

### Posiciones de Flechas

```
SINGLES (5 paneles):          DOUBLES (10 paneles):
Posición:  0  1 2  3 4           0  1 2  3 4  5  6 7  8 9
           ↑           ↑           ↑           ↑
         izquierda   derecha     izquierda   derecha
           P1          P1          P1          P2
```

### Downpress vs Hold

- `1` = downpress (nota normal)
- `2` = inicio de hold
- `3` = fin de hold
- `4` = hold activo (mientra sostienes)

---

## Sistema de Detección de Skills

### Run (Carrera)

**Archivo:** `skills.py` línea 145-182

**Definición:** Una secuencia de notas donde:
- Cada nota tiene **un solo pie** (`len(set(limb_annots[i])) == 1`)
- Los pies **alternan** (`set(limb_annots[i]) != set(limb_annots[j])`)
- El **ritmo es consistente** (`ts[i] ≈ ts[j]`)
- Mínimo 7 notas (`MIN_RUN_LEN = 7`)

**Lógica:**

```python
def run(cs: ChartStruct) -> None:
    # Itera sobre pares de líneas consecutivas
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            set(limb_annots[i]) != set(limb_annots[j]),  # pies diferentes
            len(set(limb_annots[i])) == 1,              # 1 pie por línea
            len(set(limb_annots[j])) == 1,              # 1 pie por línea
            '1' in lines[j],                            # tiene nota
            math.isclose(ts[i], ts[j])                  # mismo ritmo
        ]
        if all(crits):
            idxs.add(j)  # marca la nota j como parte del run
```

**Problema potencial:** Un `00111` con `l,r,l` en posiciones 2,3,4 podría marcarse incorrectamente porque:
- Las líneas `00111` → `00111` son **iguales**
- `num_downpress == 1` es **falso** (hay 3 downpresses)

Espera, no. Veamos de nuevo...

### Jack

**Archivo:** `skills.py` línea 564-581

**Definición:** Misma nota, mismo pie, rápida:
- Líneas **iguales** (`lines[j] == lines[i]`)
- **Un solo downpress** (`num_downpress == 1`)
- **Mismo pie** (`limb_annots[i] == limb_annots[j]`)
- Ritmo rápido (`ts[j]` - cualquier valor positivo cuenta)

```python
def jack(cs: ChartStruct) -> None:
    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            lines[j] == lines[i],              # misma línea
            notelines.num_downpress(lines[i]) == 1,  # una sola nota
            limb_annots[i] == limb_annots[j],  # mismo pie
            ts[j]                               # hay ritmo (no importa el valor)
        ]
        res.append(all(crits))
```

### Footswitch

**Archivo:** `skills.py` línea 584-599

**Definición actual:** Misma nota, pies alternados:
- Líneas **iguales** (`lines[j] == lines[i]`)
- **Un solo downpress** (`num_downpress == 1`)
- **Pies diferentes** (`limb_annots[i] != limb_annots[j]`)

```python
def footswitch(cs: ChartStruct) -> None:
    res = [False]
    for i, j in itertools.pairwise(range(len(df))):
        crits = [
            lines[j] == lines[i],                  # misma posición de flecha
            notelines.num_downpress(lines[i]) == 1,    # solo una nota
            limb_annots[i] != limb_annots[j],      # pies diferentes
        ]
        res.append(all(crits))
```

**⚠️ PROBLEMA IDENTIFICADO:** Esta función solo verifica que las líneas sean **iguales** (`lines[j] == lines[i]`), pero **NO verifica que sea la MISMA posición de flecha** en términos de `arrow_pos`. 

Si tienes:
```
time 0.0: line = `00111` (flechas en pos 2,3,4), limb = "lr"
time 0.5: line = `00111` (flechas en pos 2,3,4), limb = "lr"
```

Esto NO es footswitch porque hay 3 notas, no 1. Pero si tienes:
```
time 0.0: line = `00100` (flecha en pos 2), limb = "l"
time 0.5: line = `00100` (flecha en pos 2), limb = "r"
```

Esto SÍ es footswitch - la misma posición (pos 2) alternando entre pies.

**Mejora necesaria:** El código actual compara líneas completas, pero no tracking de **cuál flecha específica** alterna. Para triples como `00111` con `l,r,l`:
- Línea 1: `00111` → annotation `l,r,l` (3 notas)
- Línea 2: `00111` → annotation `l,r,l` (3 notas)

Esto **no debería** ser footswitch. Esas son 3 notas simultáneas, no una nota alternando.

### Bracket

**Archivo:** `skills.py` línea 57-67, 249-256

**Definición:** Una línea donde dos notas pueden ser ejecutadas con un solo pie.

```python
def has_bracket(line: str, limb_annot: str) -> bool:
    # Cuenta cuántos 'l' y cuántos 'r' hay en la annotation
    # Si hay >= 2 del mismo pie, puede ser bracket
    if limb_annot.count('l') < 2 and limb_annot.count('r') < 2:
        return False
    arrow_positions = [i for i, s in enumerate(line) if s != '0']
    if len(arrow_positions) < 2:
        return False
    # multihit_to_valid_feet() retorna qué combinaciones son válidas
    valid_limbs = notelines.multihit_to_valid_feet(arrow_positions)
    mapper = {'l': 0, 'r': 1, 'e': 0, 'h': 0}
    return tuple(mapper[l] for l in limb_annot) in valid_limbs
```

**Ejemplo:** Si tienes `10100` (flechas en pos 0 y 2) con annotation `ll`:
- `mapper['l'] = 0`, así `tuple(mapper[l] for l in 'll') = (0, 0)`
- `[0, 0]` está en `valid_limbs` para esas posiciones → es bracket

### Staggered Bracket

**Archivo:** `skills.py` línea 259-277

**Definición:** Dos líneas donde las notas, cuando se fusionan, forman un bracket.

```python
def staggered_bracket(line1: str, line2: str) -> bool:
    # Fusiona las dos líneas: donde cualquiera tenga '1', cuenta
    f = lambda c1, c2: '1' if bool(c1 == '1' or c2 == '1') else '0'
    merged_line = ''.join([f(c1, c2) for c1, c2 in zip(line1, line2)])
    return line_is_bracketable(merged_line)
```

Si line1 = `10000` y line2 = `00100`, el merged sería `10100` que es bracketable.

### Drill

**Archivo:** `skills.py` línea 91-142

**Definición:** 
- Dos líneas iniciales con pies alternando
- Líneas siguientes que repiten las primeras dos
- Ritmo consistente

```python
def drills(cs: ChartStruct) -> None:
    i, j = 0, 1
    while j < len(df):
        crits = [
            '1' in lines[i],                    # línea i tiene nota
            '1' in lines[j],                    # línea j tiene nota
            set(limb_annots[i]) != set(limb_annots[j]),  # pies alternan
            len(set(limb_annots[i])) == 1,      # solo 1 pie en i
            len(set(limb_annots[j])) == 1,      # solo 1 pie en j
        ]
        if all(crits):
            # Extiende el drill buscando repeticiones
            k = j + 1
            while k < len(df):
                if (k - i) % 2 == 0:
                    same_as = lines[k] == lines[i]  # even: misma que i
                else:
                    same_as = lines[k] == lines[j]  # odd: misma que j
                consistent_rhythm = math.isclose(ts[k], ts[j])
                if same_as and consistent_rhythm:
                    k += 1
                else:
                    break
            if k - i >= MIN_DRILL_LEN:  # mínimo 5 notas
                for idx in range(i, k):
                    drill_idxs.add(idx)
```

---

## Sistema de Razonamiento de Patrones

El sistema en `reasoning/reasoners.py` usa **LimbReusePattern** para detectar y ejecutar runs.

### LimbReusePattern

**Archivo:** `reasoners.py` línea 26-70

```python
class LimbReusePattern:
    def __init__(self, downpress_idxs: list[int], limb_pattern: list[LimbUse]):
        # downpress_idxs: índices de downpresses en el chart
        # limb_pattern: lista de LimbUse.alternate o LimbUse.same
        #    para cada par de downpresses consecutivos
        pass

    def check(self, downpress_limbs: list[int | str]) -> tuple[bool, any]:
        # Verifica si el patrón de limbs coincide con el esperado
        # alternate: pies diferentes
        # same: mismo pie
        pass

    def fill_limbs(self, starting_limb: str) -> NDArray:
        # "Llena" los limbs desde un pie inicial
        # Si limb_pattern = [alternate, same, alternate]
        # y starting_limb = 'left'
        # → [0, 1, 1, 0] (izq, alternar a der, mismo der, alternar a izq)
        pass
```

**Ejemplo:**
```
downpress_idxs = [0, 1, 2, 3, 4]
limb_pattern = [alternate, alternate, same, alternate]

Si starts with left (0):
→ [0, 1, 0, 0, 1] = l, r, l, l, r

Si starts with right (1):
→ [1, 0, 1, 1, 0] = r, l, r, r, l
```

### PatternReasoner

**Archivo:** `reasoners.py` línea 72-429

El `PatternReasoner` hace:

1. **Anota el ChartStruct** con columnas adicionales:
   - `__time since prev downpress`
   - `__time to next downpress`
   - `__line repeats previous downpress line`
   - `__line repeats next downpress line`
   - `__num downpresses`
   - `__single hold ends immediately`

2. **Encuentra runs** usando `find_runs()`:
   - Busca secuencias de notas que cumplan criterios de tempo
   - Parametros configurables via `args`:
     ```python
     MIN_TIME_SINCE = 1/13      # ~77ms mínimo entre notas
     MAX_TIME_SINCE = 1/2.5     # ~400ms máximo entre notas
     MIN_RUN_LENGTH = 5          # mínimo 5 notas
     ```

3. **Nombra runs** con `LimbReusePattern`:
   - Cada run tiene un patrón de alternancia/same
   - `__line repeats next downpress line` indica si la siguiente línea es igual (same) o diferente (alternate)

4. **Decide limbs** para cada run:
   - Prueba empezar con pie izq o pie der
   - Usa `pattern_store.score_run()` para ver cuál inicio tiene mejor "score"
   - Retorna None si no puede decidirse

### find_runs() - Lógica Detallada

**Archivo:** `reasoners.py` línea 368-429

```python
def find_runs(self) -> list[LimbReusePattern]:
    runs = []
    curr_run = None
    downpress_df = df[df['__num downpresses'] > 0]
    
    for row_idx, row in downpress_df.iterrows():
        if curr_run is None:
            curr_run = [row_idx]
        else:
            if self.is_in_run(df.iloc[curr_run[0]], row):
                curr_run.append(row_idx)
            else:
                # Evalúa si el run es largo enough
                if len(curr_run) >= self.MIN_RUN_LENGTH:
                    runs.append(curr_run)
                curr_run = [row_idx]
    
    # Merge runs que están pegados
    while (merged_runs := self.merge(runs)) != runs:
        runs = merged_runs
    
    # Convierte runs a LimbReusePatterns
    for run in runs:
        # limb_pattern de cada par de líneas consecutivas
        lp = [rnd_map[x] for x in repeats_next_downpress.iloc[run[:-1]]]
        # rnd_map = {True: LimbUse.same, False: LimbUse.alternate}
```

**is_in_run() - Criterios:**
```python
def is_in_run(self, start_row, query_row) -> bool:
    return all([
        start_row['__time since prev downpress'] >= MIN_TIME_SINCE,
        query_row['__time since prev downpress'] >= MIN_TIME_SINCE,
        query_row['__time since prev downpress'] < MAX_TIME_SINCE,
        query_row['__time to next downpress'] >= MIN_TIME_SINCE,
        notelines.num_downpress(start_line) == 1,
        notelines.num_downpress(query_line) == 1,
        '4' not in start_line,  # no hold activo
        '3' not in start_line,  # no fin de hold
        '4' not in query_line,
        '3' not in query_line,
        not jack_on_center_panel,  # jack en centro puede ser footswitch
        # ... más checks de downpress válido
    ])
```

---

## Problemas Conocidos y Mejoras Posibles

### 1. Footswitch no detecta posición específica de flecha

**Problema actual:**
La función `footswitch()` compara líneas completas (`lines[j] == lines[i]`), pero no trackea **cuál flecha específica** alterna.

**Escenario problemático:**
```
time 0.0: line = `00111`, limb = "lrl"  (pos 2,3,4)
time 0.5: line = `00111`, limb = "lrl"  (pos 2,3,4)
```

Esto tiene 3 notas simultáneas en cada línea. No es footswitch.

Pero si tienes:
```
time 0.0: line = `00100`, limb = "l"    (solo pos 2)
time 0.5: line = `00100`, limb = "r"    (solo pos 2)
```

Esto SÍ es footswitch - la misma posición (pos 2) alternando.

**Mejora sugerida:**
Para cada par de líneas consecutivas que:
- Tienen `num_downpress == 1`
- Tienen limbs diferentes

Verificar que la posición de la flecha (`arrow_pos`) sea la misma en ambas líneas.

### 2. Footswitch depende de cadencia

**Problema actual:**
El código NO tiene en cuenta la cadencia/tempo para distinguir entre:
- **Run rápido** → no debería llamarse footswitch
- **Footswitch** → notas más relajadas, alternando mismo pie

**Escenario problemático:**
```
8th notes a 180 BPM:
time 0.0: pos 2 → l
time 0.33: pos 2 → r
time 0.67: pos 2 → l
time 1.0: pos 2 → r
```

Esto se siente como un run, no como footswitch.

**Mejora sugerida:**
Añadir umbral de tiempo mínimo/máximo entre notas para clasificar como footswitch:
- `MIN_TIME_SINCE` para runs (más rápido)
- Rango diferente para footswitch (más relajado)

### 3. Bracket y Twist pueden confundirse

En triples como `00111` con `l,r,l`:
- Podría ser bracket (dos notas ejecutables con un pie)
- Podría ser un patrón de twist

El código actual depende de la limb annotation para decidir, pero si la annotation está mal, el bracket detection también estará mal.

---

## Arquitectura del Proyecto

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PRINCIPAL                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  .ssc (StepMania) ──► SongSSC/StepchartSSC ──► ChartStruct          │
│                                               │                     │
│                    ┌──────────────────────────┼───────────────────┐  │
│                    ▼                          ▼                   ▼  │
│           Anotación de Skills      Segmentación      Predicción de   │
│           (skills.py)             (segment.py)     Extremidades     │
│                    │                          │             (ml/)   │
│                    ▼                          ▼                   ▼  │
│           25+ columnas bool      list[Section]    Limb predictions │
│                    │                          │                   │  │
│                    └──────────────────────────┼───────────────────┘  │
│                                               ▼                    │
│                                    Predicción de Dificultad        │
│                                    (difficulty/)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE PREDICCIÓN ML                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ChartStruct + Limb Annotations vacías                               │
│         │                                                             │
│         ▼                                                             │
│  PatternReasoner.propose_limbs()                                    │
│         │                                                             │
│         ├── find_runs() → LimbReusePatterns                         │
│         ├── decide_limbs_for_pattern() → fill_limbs()              │
│         └── Score con pattern_store.score_run()                     │
│         │                                                             │
│         ▼                                                             │
│  Tactician (post-processing)                                         │
│         │                                                             │
│         ├── initial_predict() → modelo LightGBM                     │
│         ├── enforce_arrow_after_hold_release()                     │
│         ├── flip_labels_by_score()                                  │
│         ├── flip_jack_sections()                                    │
│         ├── beam_search()                                          │
│         ├── fix_double_doublestep()                                │
│         └── detect_impossible_multihit()                            │
│         │                                                             │
│         ▼                                                             │
│  ChartStruct + Limb Annotations completas                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Instalación

```bash
pip install -e .
```

### Dependencias

```bash
pip install lightgbm ruptures pandas numpy loguru tqdm scipy scikit-learn
```

---

## Uso para Análisis de Skills

### Cargar y detectar todos los skills

```python
from piu_annotate.formats.sscfile import SongSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills

song = SongSSC.from_file('song.ssc')
stepchart = song.get_stepchart(difficulty=15)
cs = ChartStruct.from_stepchart_ssc(stepchart)

annotate_skills(cs)  # Detecta todos los skills

# Ver results
print(cs.df[['Time', 'Line', 'Limb annotation', '__run', '__footswitch', '__jack']].head(20))
```

### Detectar skills específicos

```python
# Solo drill y run
from piu_annotate.segment.skills import drills, run

drills(cs)  # Añade columna __drill
run(cs)     # Añade columna __run
```

### Ver skill stats

```python
skill_cols = [c for c in cs.df.columns if c.startswith('__') and cs.df[c].dtype == bool]
for col in skill_cols:
    count = cs.df[col].sum()
    if count > 0:
        print(f"{col}: {count} líneas")
```

---

## Formato de Archivo

### ChartStruct CSV

```csv
Beat,Time,Line,Line with active holds,Limb annotation,Metadata
0.0,0.0,`10000,`10000,l,
0.5,0.5,`01000,`01000,r,
1.0,1.0,`00100,`00100,?,
...
```

### JSON de Visualización

```json
[
  [[arrow_pos, time, limb], ...],    // arrow_arts
  [[arrow_pos, start, end, limb], ...], // hold_arts
  {metadata}
]
```

---

## Licencia

MIT