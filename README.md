# piu-annotate

**Sistema de anotación de extremidades y predicción de dificultad para charts de Pump It Up**

Analiza charts del juego de ritmo Pump It Up (PIU), predice qué extremidad (pie izquierdo, pie derecho, mano) debe usar el jugador para cada flecha, segmenta charts en secciones de dificultad, y predice ratings de dificultad.

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
│           (drills, runs, twists,   (ruptures CPD)    Extremidades     │
│            brackets, jacks, etc.)                      │             │
│                    │                          ┌──────┴──────┐       │
│                    │                          ▼             ▼       │
│                    │                   Section List   Limb Predictions│
│                    │                          │             │       │
│                    └──────────────────────────┼─────────────┘       │
│                                               ▼                    │
│                                    Predicción de Dificultad        │
│                                    (Segment/Stepchart Models)      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Estructura de Directorios

```
piu-annotate/
├── piu_annotate/              # Paquete Python principal
│   ├── formats/                # Manejo de formatos de archivo
│   │   ├── chart.py           # ChartStruct - representación primary
│   │   ├── sscfile.py         # Parser de archivos .ssc
│   │   ├── ssc_to_chartstruct.py
│   │   ├── nps.py             # Cálculos de notas por segundo
│   │   ├── notelines.py       # Utilidades de parsing de líneas
│   │   ├── jsplot.py          # Formato para Chart.js
│   │   ├── arroweclipse.py    # Formato ArrowEclipse JSON
│   │   └── limbchecks.py      # Validación de annotaciones
│   │
│   ├── ml/                    # Machine Learning para predicción de extremidades
│   │   ├── predictor.py       # Pipeline principal de predicción
│   │   ├── models.py          # ModelWrapper, LGBModel (LightGBM)
│   │   ├── featurizers.py     # ChartStructFeaturizer
│   │   ├── datapoints.py      # ArrowDataPoint representation
│   │   └── tactics.py          # Tactician - lógica táctica post-ML
│   │
│   ├── segment/               # Segmentación de charts
│   │   ├── segment.py         # Segmenter (ruptures-based CPD)
│   │   ├── segment_breaks.py  # Detección de puntos de ruptura
│   │   └── skills.py          # Anotación de skills (25+ tipos)
│   │
│   ├── reasoning/             # Razonamiento basado en patrones
│   │   ├── reasoners.py       # PatternReasoner, LimbReusePattern
│   │   └── pattern_store.py   # Almacenamiento y matching de patrones
│   │
│   ├── difficulty/            # Predicción de dificultad
│   │   ├── models.py          # Modelos LightGBM para difficulty
│   │   ├── featurizers.py     # Feature extraction para difficulty
│   │   ├── travel.py           # Cálculo de travel del pie
│   │   └── utils.py
│   │
│   └── crawl.py               # Utilidades de crawling de archivos SSC
│
├── cli/                       # Scripts de línea de comandos
│   ├── ingest/                # Ingesta de datos
│   ├── limbuse/              # Predicción de extremidades
│   ├── segment/              # Segmentación de charts
│   ├── difficulty/           # Predicción de dificultad
│   ├── chartjson/            # Generación de JSON para visualización
│   ├── display_metadata/     # Análisis de metadata
│   ├── analysis/             # Scripts de análisis
│   ├── debug/                # Herramientas de debugging
│   └── page_content/         # Generación de contenido web
│
├── jupyter/                   # Notebooks Jupyter para análisis
└── artifacts/                # Datos y modelos (no incluido en git)
```

## Conceptos Clave

### ChartStruct

Estructura de datos principal - un DataFrame con una fila por "línea" (beat/timestamp).

**Columnas:**
- `Beat`: Número de beat
- `Time`: Tiempo en segundos
- `Line`: String como `` `10100` (6 chars para singles, 11 para doubles)
  - `0` = sin nota, `1` = flecha, `2` = inicio hold, `3` = fin hold
- `Line with active holds`: Incluye `4` para holds activos
- `Limb annotation`: String concatenado de:
  - `l` = pie izquierdo
  - `r` = pie derecho
  - `e` = cualquier pie
  - `h` = cualquier mano
  - `?` = desconocido

### Skills Anotados (25+ tipos)

**Básicos:** drill, run, anchor_run, jack, footswitch, bracket, staggered_bracket, doublestep, hands, jump

**Twists:** twist_90, twist_over90, twist_close, twist_far

**Posición-specific:** side3_singles, mid4_doubles, mid6_doubles, split

**Holds:** hold_footswitch, hold_footslide

**Otros:** stair5, stair10, yog_walk, cross_pad_transition, coop_pad_transition

### Pipeline de Predicción de Extremidades

```
1. PatternReasoner.propose_limbs() → encuentra runs, crea LimbReusePatterns
2. Tactician.initial_predict() → usa modelo arrow_to_limb
3. enforce_arrow_after_hold_release() → restricción física
4. flip_labels_by_score() → optimiza por score
5. flip_jack_sections() → repara jacks
6. beam_search() → exploración de K-mejores por iteración
7. fix_double_doublestep() → repara patrones impossibles
8. detect_impossible_multihit() → repara brackets impossibles
```

## Instalación

```bash
pip install -e .
```

### Dependencias

```bash
pip install lightgbm ruptures pandas numpy loguru tqdm scipy scikit-learn
```

## Uso Rápido

### 1. Parsear archivo SSC a ChartStruct

```python
from piu_annotate.formats.sscfile import SongSSC
from piu_annotate.formats.chart import ChartStruct

song = SongSSC.from_file('song.ssc')
stepchart = song.get_stepchart(difficulty=7, play_style=' singles')
cs = ChartStruct.from_stepchart_ssc(stepchart)
```

### 2. Predecir extremidades

```python
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict

model_suite = ModelSuite.load('models/')
cs, fcs, pred_limbs = predict(cs, model_suite, verbose=True)
cs.add_limb_annotations(fcs.pred_coords, pred_limbs, 'Predicted limbs')
```

### 3. Segmentar chart

```python
from piu_annotate.segment.segment import Segmenter

segmenter = Segmenter()
sections = segmenter.segmentation(cs)
```

### 4. Predecir dificultad

```python
from piu_annotate.difficulty.models import DifficultySegmentModelPredictor

predictor = DifficultySegmentModelPredictor.load('models/')
for section in sections:
    diff = predictor.predict(section)
```

## Scripts CLI

### Ingesta

```bash
python -m cli.ingest.crawl_piu_simfiles --input-dir /path/to/sscs
python -m cli.ingest.make_chartstructs --input-dir /path/to/sscs --output-dir ./chartstructs/
```

### Predicción de extremidades

```bash
python -m cli.limbuse.train_lgbm --dataset /path/to/dataset
python -m cli.limbuse.predict_limbs --input ./chartstructs/ --models ./models/ --output ./predicted/
```

### Segmentación

```bash
python -m cli.segment.segment_charts --input ./chartstructs/ --output ./segments/
```

### Dificultad

```bash
python -m cli.difficulty.train_difficulty_predictor --dataset /path/to/data
python -m cli.difficulty.annotate_segment_difficulty_with_segment_model --input ./segments/
```

## Para Agentes IA

### APIs Principales para Agentes

#### 1. Parsing y Carga

```python
# Cargar song SSC
from piu_annotate.formats.sscfile import SongSSC
song = SongSSC.from_file(path)  # → SongSSC

# Obtener stepchart específico
stepchart = song.get_stepchart(difficulty=12, play_style='singles')  # → StepchartSSC

# Convertir a ChartStruct
cs = ChartStruct.from_stepchart_ssc(stepchart)  # → ChartStruct
```

#### 2. Acceso a Datos del ChartStruct

```python
# DataFrame principal
df = cs.df

# Líneas del chart (sin `)
lines = cs.get_lines()  # list[str]

# Coordenadas de flechas
arrow_coords = cs.get_arrow_coordinates()  # list[ArrowCoordinate]

# Coordenadas para predicción (solo downpresses)
pred_coords = cs.get_prediction_coordinates()  # list[ArrowCoordinate]

# Metadata del chart
metadata = cs.metadata  # dict
level = cs.get_chart_level()  # int
sord = cs.singles_or_doubles()  # 'singles' | 'doubles'
```

#### 3. Anotación de Skills

```python
from piu_annotate.segment.skills import annotate_skills

annotate_skills(cs, skill_names=['drill', 'run', 'bracket', 'twist_90'])
# Añade columnas booleanas al df: 'drill', 'run', 'bracket', etc.
```

#### 4. Segmentación

```python
from piu_annotate.segment.segment import Segmenter

segmenter = Segmenter()
sections = segmenter.segmentation(cs)  # → list[Section]

for section in sections:
    print(f"Section: {section.start_time:.2f} - {section.end_time:.2f}")
    print(f"  Difficulty: {section.difficulty}")
```

#### 5. Predicción de Extremidades

```python
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict

model_suite = ModelSuite.load(model_path)
cs, fcs, pred_limbs = predict(cs, model_suite, verbose=False)

# Añadir predicciones al ChartStruct
cs.add_limb_annotations(fcs.pred_coords, pred_limbs, 'Predicted limbs')
```

#### 6. Predicción de Dificultad

```python
from piu_annotate.difficulty.models import DifficultySegmentModelPredictor

predictor = DifficultySegmentModelPredictor.load(model_path)
difficulty = predictor.predict(section)  # → float
```

#### 7. Exportar a JSON para Visualización

```python
from piu_annotate.formats.jsplot import ChartJsStruct

cjs = ChartJsStruct.from_chartstruct(cs)
cjs.to_json('chart.json')
```

### Patrones Comunes para Agentes

#### Procesar un chart completo

```python
def process_chart(ssc_path, model_suite_path):
    song = SongSSC.from_file(ssc_path)
    stepchart = song.stepcharts[0]  # o filtrar por difficulty
    cs = ChartStruct.from_stepchart_ssc(stepchart)

    annotate_skills(cs)

    segmenter = Segmenter()
    sections = segmenter.segmentation(cs)

    model_suite = ModelSuite.load(model_suite_path)
    cs, fcs, pred_limbs = predict(cs, model_suite)

    cs.add_limb_annotations(fcs.pred_coords, pred_limbs, 'Predicted limbs')

    return cs, sections
```

#### Buscar charts por nivel

```python
def find_charts_by_level(chartstruct_dir, level):
    import os
    matches = []
    for fn in os.listdir(chartstruct_dir):
        if fn.startswith(f'S{level}') or fn.startswith(f'D{level}'):
            matches.append(os.path.join(chartstruct_dir, fn))
    return matches
```

## Formato de Archivo

### ChartStruct CSV

```csv
Beat,Time,Line,Line with active holds,Limb annotation,Metadata
0.0,0.0,`10000,`10000,l,
0.5,0.5,`01000,`01000,r,
...
```

### JSON de Visualización

```json
{
  "arrow_arts": [
    {"arrow_pos": 0, "time": 0.0, "limb": "l"},
    {"arrow_pos": 1, "time": 0.5, "limb": "r"}
  ],
  "hold_arts": [
    {"arrow_pos": 0, "start_time": 1.0, "end_time": 2.0, "limb": "l"}
  ]
}
```

## Desarrollo

### Ejecutar tests

```bash
pytest tests/
```

### Linting

```bash
ruff check piu_annotate/
black piu_annotate/
```

## Licencia

MIT