# Plan de implementación: Transformer (MLX) + fix de brackets pie derecho

> **Audiencia:** un agente que vaya a ejecutar este plan paso a paso, sin contexto previo.
> **Objetivo doble:**
> 1. Diagnosticar y reducir los errores específicos de brackets con pie derecho en singles (techo del modelo LightGBM actual: 75.4 %).
> 2. Migrar el suite a un Transformer (MLX) que procese el chart como **secuencia temporal completa**, no como ventana fija — habilita memoria de orientación y debe llevar singles a >90 %.
>
> **Hardware asumido:** Apple Silicon (M-series), 24 GB RAM. Todo el entrenamiento corre con MLX (CPU+GPU unified memory). NO se va a usar PyTorch ni CUDA.
>
> **Tiempo total estimado:** 8–14 h de trabajo del agente (incluye varias rondas de entrenamiento). Las rondas largas pueden mandarse en background.

---

## 0. Estado del mundo antes de empezar

Lee y verifica que tienes esto en disco antes de tocar nada:

| Archivo / dir | Qué contiene | Por qué importa |
|---|---|---|
| `artifacts/manual-chartstructs/visss-120524-eaware/` | ~4261 CSV de training con `e` en centros ambiguos | Dataset base. NO regenerar. |
| `artifacts/models/visss/{singles,doubles}-*.txt` | 8 sub-modelos LightGBM (baseline) | Baseline para comparar contra el Transformer. NO borrar. |
| `piu_annotate/ml/featurizers.py` | `ChartStructFeaturizer` con `featurize_arrows_with_context()` | Reutilizamos para los features por arrow; NO reescribir. |
| `piu_annotate/ml/mlx_architecture.py` | `LimbTransformer` actual — sliding window, sin masking real | **Lo vamos a reescribir.** Está documentado como stub. |
| `cli/limbuse/train_mlx.py` | Loop de training MLX — 3 epochs, batch 64, secuencia de longitud 1 | **Lo vamos a reescribir.** No procesa el chart como secuencia. |
| `piu_annotate/ml/predictor.py` | Pipeline de inferencia (Tactician + PatternReasoner) | Hay que añadirle un branch para usar el Transformer. |
| `piu_annotate/ml/tactics.py` | 705 LOC, contiene reglas físicas | Vamos a añadir `body_rotation_penalty`. |
| `scripts/benchmark_annotations.py` | Compara `processed_db` vs vis-ss | Métrica oficial. |

Comando de verificación:
```bash
cd /Users/rodrigo/dev/piu/piu-annotate_to_label_piu
ls artifacts/manual-chartstructs/visss-120524-eaware/ | wc -l   # ~4261
ls artifacts/models/visss/ | wc -l                              # 8
python3 -c "import mlx.core as mx; print(mx.default_device())"  # debe imprimir Device(gpu, 0) o cpu
```

Si MLX no está, instálalo con `pip install mlx mlx-data` (NO `mlx[cuda]`).

---

## Fase A — Diagnóstico de errores de bracket pie-derecho (1–2 h)

**Hipótesis del usuario:** el modelo se equivoca en brackets con pie derecho de forma sistemática. Antes de cambiar arquitectura, hay que **medir** dónde y por qué falla, así sabemos contra qué evaluar el Transformer.

### A.1 — Crear el script de diagnóstico

Crea `scripts/diagnose_bracket_errors.py`. Tiene que:

1. Cargar todos los CSV de `artifacts/processed_db/` (predicciones actuales) y los respectivos de `artifacts/manual-chartstructs/visss-120524-eaware/` (ground truth) — match por nombre de archivo.
2. Para cada chart, recorrer fila por fila y clasificar cada **bracket** (línea con `num_downpress_in_line >= 2` y `line_is_bracketable=True`) en:
   - `LL_correct` / `LL_wrong` (bracket pie izquierdo correcto / incorrecto)
   - `RR_correct` / `RR_wrong` (bracket pie derecho)
   - `LR_correct` / `LR_wrong` (jump bilateral / mixto)
3. Para cada bracket equivocado, guardar: `(chart_name, row_idx, line, gt_annot, pred_annot, level, prev_orientation)` donde `prev_orientation` es la última anotación no-`e` antes del error (mira hasta 8 filas atrás).
4. Imprimir tabla resumen:
   ```
   Bracket type   N      Acc       FP→LL   FP→RR   FP→LR
   LL (gt)        12345  92.1%     —       7.2%    0.7%
   RR (gt)         8901  78.3%     19.8%   —       1.9%
   LR (gt)         3456  88.4%     5.5%    6.1%    —
   ```
5. Guardar los 100 peores casos de `RR_wrong` en `artifacts/debug/rr_bracket_errors.csv` con columnas `chart, row, line, gt, pred, level, context_before` para inspección manual.

**Decisión clave:** si `RR_acc` ≥ `LL_acc - 3pp`, la hipótesis del usuario es falsa y el problema NO es asimetría L/R, sino general. Anota el resultado en `progress.md` antes de continuar. Si `RR_acc` < `LL_acc - 3pp`, sigue con A.2.

### A.2 — Inspección manual

Lee `artifacts/debug/rr_bracket_errors.csv` y agrupa los errores por **patrón de línea** (`line` columna). Las hipótesis a falsificar:

- **H1: Asimetría de features.** El featurizer produce features espacialmente sesgadas (e.g. coordenadas absolutas de pad sin reflejar). Verifica leyendo `piu_annotate/ml/datapoints.py` cómo se construye `ArrowDataPoint.x/y` y `prev_pc_idxs`. Si los features tratan a panel 0 (DL) y panel 4 (DR) de forma simétrica, H1 es falsa.
- **H2: Sesgo de dataset.** vis-ss tiene más LL-brackets que RR-brackets, y el modelo aprende la prior. Cuenta en el ground-truth: `for csv in visss-120524-eaware: count rows where annot in {ll, rr}`. Si `count(ll) > 1.3 * count(rr)`, H2 es real → considera **data augmentation por reflexión horizontal** (panel 0↔4, 1↔3, 5↔9, 6↔8 + swap l↔r en `Limb annotation`) durante training. Esto es independiente de la arquitectura y debe añadirse SIEMPRE.
- **H3: Contexto insuficiente.** Los brackets RR fallan más cuando el orientation previo está fuera de la ventana de 20 arrows del featurizer. Esto se valida solo cuando entres con el Transformer (Fase B+) que tiene contexto largo.

### A.3 — Augmentation por reflexión (independiente de Transformer)

**Implementa esto incluso si vas a entrenar Transformer**, porque también beneficia al baseline LightGBM y al benchmark.

Crea `piu_annotate/formats/mirror.py` con:

```python
PANEL_MIRROR_SINGLES = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
PANEL_MIRROR_DOUBLES = {0:9, 1:8, 2:7, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1, 9:0}

def mirror_line(line: str, sd: str) -> str:
    """ Refleja una línea de chart horizontalmente. Mantiene `s, holds, etc. """
    ...

def mirror_limb_annot(annot: str) -> str:
    """ Reverse string + swap l<->r (e queda igual). """
    ...

def mirror_chartstruct(cs: ChartStruct) -> ChartStruct:
    """ Devuelve copia espejada. NO modifica el original. """
    ...
```

Tests obligatorios en `tests/test_mirror.py`:
- `mirror_line('10001', 'singles') == '10001'` (simétrico → igual)
- `mirror_line('11000', 'singles') == '00011'`
- `mirror_limb_annot('llr') == 'lrr'` (reversa + swap)
- `mirror_limb_annot('le') == 'er'` (reversa + swap, e invariante)
- `mirror_chartstruct(mirror_chartstruct(cs))` produce CSV idéntico al original (round-trip).

El espejado se aplica **on-the-fly durante training** con prob 0.5, NO se materializa a disco — así no se duplica el dataset físico.

**Criterio de salida de Fase A:** tienes la tabla de A.1 y o bien (a) el augmentation listo y testeado, o (b) evidencia de que H1/H2 son falsas. Documenta en `docs/progress.md` la sección **A.X bracket diagnostic** con un párrafo.

---

## Fase B — Reescribir el Transformer (chart-as-sequence) (3–4 h)

El Transformer actual (`mlx_architecture.py`) procesa cada arrow con su ventana ya aplanada — eso es exactamente lo que hace LightGBM, sin la ventaja de attention global. **Hay que rehacerlo para que la unidad de entrada sea el chart entero (truncado/segmentado).**

### B.1 — Diseño

| Decisión | Valor | Razón |
|---|---|---|
| **Unidad de input** | Secuencia de N pred-coords del chart (N hasta 1024) | Permite attention global sobre todo el chart |
| **Truncamiento** | Charts >1024: cortar en segmentos solapados de 1024 con overlap de 128 | El 99 %ile de charts tiene <1024 pred-coords; verifícalo |
| **Features por token** | `ArrowDataPoint.to_array_categorical()` SIN ventana (solo el arrow ese) + chart-level features broadcast | El attention reemplaza la ventana fija |
| **Posicional** | Sinusoidal + un token-channel `time_since_prev_downpress` ya está en los features | Sinusoidal absoluto basta porque la posición relativa la captura el feature de tiempo |
| **Profundidad** | 4 layers, 8 heads, d_model=128, ffn=512 | Cabe holgado en 24 GB para batch 32 con seq_len 1024 |
| **Cabeza de salida** | 2 logits por token: `P(left)`, `P(right)`. La clase `e` no se predice — se mapea como antes (`e→0` en training) | Mantener compatibilidad con `LimbLabel.from_limb_annot` |
| **Pérdida** | BCE por token, **enmascarada** para padding y para tokens donde `gt==e` (esos no contribuyen a la loss) | No queremos que el modelo aprenda a predecir `e` ni penalizarlo por elegir L o R en un centro ambiguo |
| **Mask de attention** | Causal **NO** — el modelo puede ver futuro. Razón: estamos clasificando, no generando, y el orientation se deduce mejor con visión bidireccional. | |
| **Padding mask** | Sí — los tokens de padding no contribuyen a attention ni loss | |

### B.2 — Reescribir `piu_annotate/ml/mlx_architecture.py`

Reemplaza el archivo completo con esta estructura (no copies literal — tradúcelo a código real con docstrings):

```python
from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn

def sinusoidal_pos_encoding(seq_len: int, d_model: int) -> mx.array:
    """ Estándar Vaswani 2017. Devuelve (seq_len, d_model). """
    ...

class LimbSequenceTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 512,
        max_len: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 1,  # binario L/R; 1 logit + sigmoid
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = sinusoidal_pos_encoding(max_len, d_model)  # constante, no entrenable
        self.encoder = nn.TransformerEncoder(
            num_layers=n_layers,
            dims=d_model,
            num_heads=n_heads,
            mlp_dims=ffn_dim,
            checkpoint=True,  # gradient checkpointing — ahorra memoria
        )
        self.dropout = nn.Dropout(dropout)
        self.out_head = nn.Linear(d_model, n_classes)

    def __call__(
        self,
        x: mx.array,           # (B, L, D_in)
        padding_mask: mx.array # (B, L) — True donde es padding
    ) -> mx.array:             # (B, L, n_classes) logits
        B, L, _ = x.shape
        h = self.input_proj(x) + self.pos_enc[:L][None, :, :]
        h = self.dropout(h)
        # MLX TransformerEncoder accepts an additive mask of shape (L, L) o (B, n_heads, L, L)
        # Construir attention mask donde tokens con padding_mask=True son -inf:
        attn_mask = self._build_padding_mask(padding_mask)
        h = self.encoder(h, mask=attn_mask)
        return self.out_head(h)

    @staticmethod
    def _build_padding_mask(padding_mask: mx.array) -> mx.array:
        """ Devuelve mask aditivo (B, 1, 1, L) con -inf donde es pad. """
        ...
```

**Importante** sobre la API de `mlx.nn.TransformerEncoder` (verifícalo antes de codear):
- En la versión instalada, `forward` puede esperar `(x, mask)` con mask aditivo broadcasteable. Si la versión es vieja y no soporta padding mask externo, **escribe el encoder a mano** con `nn.MultiHeadAttention` por layer. Esto suma ~80 LOC pero garantiza control. Antes de decidir, corre:
  ```bash
  python3 -c "import mlx.nn as nn, inspect; print(inspect.signature(nn.TransformerEncoder.__call__))"
  ```
- Si MLX no expone `nn.TransformerEncoder` con la firma esperada, escribe la versión manual. Documenta la decisión arriba del archivo.

### B.3 — Adaptar el featurizer para devolver secuencias completas

En `piu_annotate/ml/featurizers.py`, **añade** (no reemplaces) un método nuevo:

```python
def featurize_arrows_as_sequence(self) -> tuple[NDArray, NDArray]:
    """ Returns:
        x: (N, D_per_arrow) — features SIN sliding window.
                              Solo el arrow + metadata de chart broadcast.
        valid_mask: (N,) bool — todos True (placeholder; el padding lo añade el dataloader)
    """
    pt_array = np.stack(self.pt_array)  # (N, D)
    cmf = np.tile(self.chart_metadata_features, (len(pt_array), 1))  # (N, M)
    x = np.concatenate([pt_array, cmf], axis=1)
    return x, np.ones(len(x), dtype=bool)
```

Esto reusa toda la maquinaria de `ArrowDataPoint`. **NO uses** `featurize_arrows_with_context` para el Transformer — la ventana es lo que el attention reemplaza.

### B.4 — Dataloader

Crea `piu_annotate/ml/mlx_dataset.py`. Responsabilidades:

1. Lee CSVs en streaming (no cargues los 4261 a RAM al mismo tiempo).
2. Por chart: featurize_arrows_as_sequence → `(x, mask, y)`.
3. Si `len(x) > 1024`: parte en chunks solapados de 1024 con overlap 128.
4. **Augmentation** (de Fase A.3): con prob 0.5 aplica `mirror_chartstruct` antes de featurizar.
5. Bucketing por longitud para minimizar padding: agrupa charts por `len // 64` y batchea dentro del bucket.
6. Devuelve batches `(x: B×L×D, padding_mask: B×L, y: B×L, loss_mask: B×L)` donde `loss_mask=False` en padding y en tokens con label `e`.

Tests:
- `test_chunking_overlaps_correctly`
- `test_padding_mask_aligns_with_y`
- `test_mirror_augmentation_preserves_pair_count`

### B.5 — Reescribir `cli/limbuse/train_mlx.py`

Reemplaza el archivo. Estructura:

```python
def main():
    args = parse_args()  # --singles_or_doubles, --manual_chart_struct_folder,
                          # --out_dir, --epochs (default 20), --batch_size (default 16),
                          # --lr (default 3e-4), --seed (default 0), --resume_from
    set_seed(args.seed)

    # 1. Build dataloaders (train 90% / val 10%, split por chart no por arrow)
    train_loader, val_loader, input_dim = build_loaders(...)

    # 2. Build model
    model = LimbSequenceTransformer(input_dim=input_dim, ...)
    mx.eval(model.parameters())

    # 3. Optimizer + LR schedule (warmup 1 epoch, cosine decay)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    # 4. Training loop — masked BCE
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader)
        val_metrics = evaluate(model, val_loader)
        log_metrics(epoch, val_metrics)
        if val_metrics['acc'] > best_acc:
            save_checkpoint(model, args.out_dir, tag='best')

    # 5. Save final
    save_checkpoint(model, args.out_dir, tag='final')
```

Métricas a loggear por epoch:
- Loss (train, val)
- Accuracy global (val)
- **Accuracy desglosada:** taps, jacks, triple-taps, holds, **LL-brackets, RR-brackets, jumps**
- Confusion matrix L↔R

Usa `tqdm` para progreso. Loguea con `loguru`. Guarda checkpoints en safetensors.

**Sub-modelos:** mantén la estructura de 4 sub-modelos del baseline (`arrows_to_limb`, `arrowlimbs_to_limb`, `matchnext`, `matchprev`) para que el Tactician siga funcionando. Para empezar entrena solo `arrows_to_limb` con el Transformer; los otros 3 los puedes dejar en LightGBM (interoperabilidad por ModelSuite).

### B.6 — Smoke test

Antes de entrenar 20 epochs, valida el pipeline con un mini-run:

```bash
python3 cli/limbuse/train_mlx.py \
  --singles_or_doubles singles \
  --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --out_dir artifacts/models/visss-mlx-smoke/ \
  --epochs 1 --batch_size 4 --limit_charts 50 2>&1 | tee out_mlx_smoke.log
```

**Esperado:** corre sin OOM, loss decrece, val_acc > random (~50 %). Si no, depura ANTES de lanzar el run completo.

---

## Fase C — Training completo + ablation (3–5 h, mayormente background)

### C.1 — Limpiar caché

```bash
rm -rf cli/temp/dataset-storage/*.pkl.gz 2>/dev/null
```

### C.2 — Run principal singles

```bash
python3 cli/limbuse/train_mlx.py \
  --singles_or_doubles singles \
  --manual_chart_struct_folder artifacts/manual-chartstructs/visss-120524-eaware/ \
  --out_dir artifacts/models/visss-mlx/ \
  --epochs 20 --batch_size 16 --lr 3e-4 \
  --seed 0 2>&1 | tee out_mlx_singles.log
```

**Tiempo estimado:** 90–150 min en M-series. Mándalo en background (`run_in_background=true`) y revisa cuando termine.

**Criterios de éxito:**
- val_acc global ≥ 92 % (vs 75 % LightGBM baseline)
- val_acc en RR-brackets ≥ val_acc en LL-brackets - 2pp (asimetría reducida)
- Sin overfitting destructivo: `train_acc - val_acc < 4pp` al final

Si NO se cumplen:
- val_acc global < 85 % → modelo demasiado pequeño o LR mal. Sube a `d_model=192`, `n_layers=6`, baja LR a 1e-4, re-entrena.
- Asimetría sigue → el augmentation no está activo o el dataloader la rompe. Verifica con `print(batch['was_mirrored'].mean())` en train loop, debe estar ~0.5.
- Overfit > 4pp → sube dropout a 0.2, añade weight decay 0.05, o reduce capas.

### C.3 — Run doubles

Idéntico pero con `--singles_or_doubles doubles`. Doubles tiene más features espaciales y el chart es más largo — vigila memoria. Si OOM, baja `batch_size` a 8 y sube `n_accumulation_steps` a 2 (gradient accumulation).

### C.4 — Ablation mínimo (opcional pero recomendado)

Corre 3 variantes adicionales de **singles**, 5 epochs cada una:
1. Sin mirror augmentation → mide cuánto aporta.
2. Sin pos encoding sinusoidal (solo features) → confirma que ayuda.
3. Profundidad 2 layers → confirma que 4 vale la pena.

Guarda resultados en `docs/transformer_ablation.md`.

---

## Fase D — Integración con `predict.py` (1–2 h)

### D.1 — Extender `ModelSuite`

`piu_annotate/ml/models.py` (revísalo primero, no lo he leído entero) probablemente carga 4 modelos `.txt` LightGBM por sd. Añade soporte para safetensors MLX.

API objetivo:
```python
suite = ModelSuite.load(
    folder='artifacts/models/visss-mlx/',
    sd='singles',
    backend='mlx',  # o 'lgbm', default 'lgbm'
)
suite.predict_arrows_to_limb(features) -> NDArray  # (N, 1) probabilidades
```

Si `backend='mlx'`, internamente:
- Carga el `LimbSequenceTransformer` con safetensors.
- En `predict_arrows_to_limb`, recibe **una secuencia de un chart**, hace forward, devuelve probas por arrow.
- Para los 3 sub-modelos restantes (matchnext, matchprev, arrowlimbs) cae en LightGBM si no hay safetensors equivalente.

### D.2 — Wiring en `predictor.py`

`predict()` ya recibe `model_suite`. No cambies su firma. Solo asegura que `tactics.initial_predict` y `flip_labels_by_score` funcionan con cualquier backend (deberían — el Tactician trata el modelo como caja negra que devuelve probas).

### D.3 — Inferencia en bulk

```bash
python3 cli/ingest/process_db_matches.py --model_backend mlx 2>&1 | tee out_processdb_mlx.log
```

Añade el flag `--model_backend` al script (`--model_backend {lgbm,mlx}`, default `lgbm` para no romper nada).

---

## Fase E — Body rotation penalty en Tactician (1 h)

Roadmap menciona penalizar rotaciones imposibles de cadera. Esto es independiente del Transformer y debería bajar los warnings de `impossible line with holds`.

### E.1 — Heurística

En `piu_annotate/ml/tactics.py`:

```python
def body_rotation_penalty(
    self,
    pred_limbs: NDArray,
    threshold_deg: float = 135.0,
    weight: float = 0.5,
) -> float:
    """ Suma penalización por cada par consecutivo de pred_coords donde
        el ángulo de cadera implícito (línea LF→RF) rota más de threshold_deg
        respecto al par anterior, en menos de X ms.

        Usa pos_map de difficulty.travel para coordenadas reales de los pads.
    """
    ...
```

Integra en `tactics.score()` como un término aditivo negativo. Empieza con `weight=0.3`, ajusta con grid search en val (`[0.1, 0.3, 0.5, 1.0]`).

### E.2 — Test de regresión

`tests/test_body_rotation.py`:
- Una secuencia "natural" (LF DL → RF DR → LF UL → RF UR) tiene penalty ≈ 0.
- Una secuencia "spin" (LF DR → RF DL → LF UR → RF UL) tiene penalty alto.
- La penalty NO se aplica cuando `time_since_prev_downpress > 0.5s` (giros lentos están permitidos).

---

## Fase F — Benchmark + decisión final (30 min)

```bash
python3 scripts/benchmark_annotations.py --show_worst 30 --top_diff 30 --plot \
  > benchmark_mlx_$(date +%Y%m%d).log
```

Compara contra el baseline:

| Métrica | LGBM (2026-05-04) | Target MLX |
|---|---|---|
| Tap overall | 96.5 % | ≥ 96.5 % (no regresión) |
| Jacks | 93.4 % | ≥ 94 % |
| Triple-taps | 86.9 % | ≥ 90 % |
| **Singles overall** | 75.4 % | **≥ 90 %** |
| Doubles overall | 96.5 % | ≥ 96 % |
| RR-bracket acc (de Fase A) | medido | ≥ LL_acc - 2pp |
| Holds | 97 % | ≥ 97 % |

**Si todas las métricas cumplen:** sync a piulatam con `python3 sync_to_piulatam.py`. Actualiza `docs/progress.md` con la nueva sección "Stage: Transformer Migration — Completed". Commit (no push) con mensaje `Migrate limb prediction to MLX Transformer`.

**Si singles < 90 %:** NO sincronices. Documenta el resultado, comparte el log, y para hasta que el usuario decida (puede ser que el techo real del problema esté ahí, o falte más capacidad / más data).

**Si hay regresión en doubles o holds:** problema serio — el Transformer está rompiendo algo que LightGBM resolvía. Revisa que el Tactician aún corra (impossible-multihit, hold-release enforcement). Probablemente el branch `mlx` saltó alguna corrección.

---

## Checklist final (para el agente, antes de declarar terminado)

- [ ] Fase A.1: tabla de accuracy por bracket type imprimida y guardada.
- [ ] Fase A.3: `mirror.py` + tests pasan.
- [ ] Fase B.2: `LimbSequenceTransformer` reescrito y smoke-tested.
- [ ] Fase B.4: dataloader con bucketing + augmentation + loss masking de `e`.
- [ ] Fase B.5: `train_mlx.py` reescrito, métricas desglosadas.
- [ ] Fase B.6: smoke run de 1 epoch x 50 charts pasa.
- [ ] Fase C.2: run completo singles guardado en `artifacts/models/visss-mlx/`.
- [ ] Fase C.3: run completo doubles guardado.
- [ ] Fase D.1: `ModelSuite` soporta `backend='mlx'`.
- [ ] Fase D.3: `process_db_matches.py` corre con `--model_backend mlx`.
- [ ] Fase E.1: `body_rotation_penalty` integrado en `tactics.score`.
- [ ] Fase F: benchmark corrido, tabla comparativa en `docs/progress.md`.

---

## Reglas de conducta para el agente

1. **NO borres** los modelos LightGBM ni el dataset. Son el baseline y la fuente de verdad.
2. **NO sincronices a piulatam** automáticamente. Eso lo decide el usuario al ver los benchmarks.
3. **NO hagas push a git**. Solo commits locales, y solo si todas las pruebas pasan.
4. Si una decisión técnica te bloquea (e.g. la API de MLX cambió, o el smoke test no pasa), **detente y reporta** — no improvises arquitecturas alternativas sin avisar.
5. Mantén `progress.md` actualizado al final de cada fase con un párrafo conciso (2–4 frases): qué hiciste, qué resultado dio, siguiente paso.
6. Si encuentras que la hipótesis del usuario sobre brackets RR es falsa (Fase A.1), repórtalo claramente — es un hallazgo valioso y cambia el plan.
7. Para cada archivo nuevo: tests obligatorios. Para cada archivo modificado: corre los tests existentes antes de declarar terminado.

---

## Apéndice: comandos útiles para el agente

```bash
# Verificar que MLX detecta GPU
python3 -c "import mlx.core as mx; a = mx.array([1.0]); print(mx.default_device(), a.dtype)"

# Ver tamaño de un modelo MLX guardado
python3 -c "
import mlx.core as mx
w = mx.load('artifacts/models/visss-mlx/singles-arrows_to_limb-best.safetensors')
print(sum(v.size for v in w.values()) * 4 / 1e6, 'MB FP32')"

# Inspeccionar un CSV de training
python3 -c "
import pandas as pd
df = pd.read_csv('artifacts/manual-chartstructs/visss-120524-eaware/<some>.csv')
print(df.columns.tolist()); print(df.head())"

# Memoria en uso (macOS)
vm_stat | head -5

# Lanzar training en background con logs
nohup python3 cli/limbuse/train_mlx.py [...] > out_mlx_singles.log 2>&1 &
echo $!  # PID para matar si hace falta
```
