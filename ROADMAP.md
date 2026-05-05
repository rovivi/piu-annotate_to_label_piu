# PIU Limb Annotation — Historial Completo y Roadmap

> **Proyecto:** `piu-annotate_to_label_piu`
> **Objetivo final:** Predecir qué pie (izquierdo/derecho/ambos) toca cada flecha en charts de Pump It Up con la misma precisión que el mejor anotador humano (fefemz).
> **Última actualización:** 2026-05-05

---

## PARTE 1 — Qué se hizo (historial completo)

### Etapa 0 — Baseline LightGBM (75.4% singles)

**Estado:** Completada

El primer modelo funcional para singles. Un GBDT (LightGBM) con features manuales que incluían posición de panel, timing, duración de holds, y reglas de corchetes. El modelo miraba una ventana corta de notas sin contexto de secuencia.

| Métrica | Valor |
|---|---|
| Singles val accuracy | **75.4%** |
| Doubles val accuracy | **96.5%** |
| Arquitectura | LightGBM, features manuales |
| Problema principal | No modela momentum del jugador; falla en corchetes y streams complejos de nivel alto |

**Qué se hizo en esta etapa:**
- Ingeniería de features manual (col_idx, beat_time, is_hold_head, is_bracket, foot_dist, etc.)
- Pipeline de entrenamiento LightGBM en `cli/limbuse/train_lgbm.py`
- Procesamiento completo de 637 canciones / 4,418 charts
- Sincronización al servidor piulatam
- Detección del cuello de botella: el modelo GBDT no puede superar ~75% en singles por falta de contexto de secuencia

**Conclusión:** El GBDT alcanzó su techo. Para >90% en singles, se necesita modelado de secuencia (Transformers).

---

### Etapa 1 — Primer Transformer MLX (versiones v1–v6)

**Estado:** Completada (prototipos)

Serie de experimentos exploratorios con el transformer en MLX. Se estableció el pipeline básico: featurización → caché .npz → entrenamiento con `LimbSequenceTransformer`. Se exploró el espacio de hiperparámetros.

**Modelos entrenados:** `visss-mlx`, `visss-mlx-smoke`, `visss-mlx-v2` → `visss-mlx-v6`

**Aprendizajes clave:**
- El transformer aprende secuencias mucho más rápido que GBDT
- La feature `prev_limb` (qué pie pisó la flecha anterior) es crítica
- El bug de atención bidireccional + teacher-forcing produce **data leakage**: token i puede ver token i+1, cuya `prev_limb` revela el label de token i → métricas infladas, inferencia real mala

---

### Etapa 2 — v7b: Causal Mask + Both-Mirror (88.6%)

**Estado:** Completada

**Modelo:** `artifacts/models/visss-mlx-v7b/` — 5M params, d_model=256, 6 capas

El salto de diseño más importante hasta la fecha: corrección del bug de atención bidireccional.

| Cambio | Impacto |
|---|---|
| Causal mask (upper-triangular = −∞) | Elimina data leakage. Token i solo puede ver posiciones 0..i |
| Both-mirror augmentation | L/R simétrico. Cada chart se entrena con original + espejo por época |
| prev_limb one-hot como feature (4-dim) | Contexto del último pie pisado como input del modelo |

**Arquitectura v7b:**
```
Input: N arrows × 22 features (18 arrow + 4 prev_limb one-hot)
→ LayerNorm
→ Linear(22 → 256)
→ Sinusoidal PE
→ Dropout(0.1)
→ TransformerEncoder × 6 capas [n_heads=8, ffn_dim=1024]
→ LayerNorm
→ OutputHead [256 → 128 → GELU → 3 clases]
```

| Métrica | Valor |
|---|---|
| Best val accuracy | **88.6%** (epoch 30/30) |
| AR gap | Bajo pero presente |
| Params | 5M |
| Tiempo de entrenamiento | ~53 min |
| Problema restante | Sin scheduled sampling → exposure bias en inferencia |

---

### Etapa 3 — v8: Scheduled Sampling + Modelo Grande (91.4%) ★ ESTADO ACTUAL

**Estado:** Completada — **mejor modelo a la fecha**

**Modelo:** `artifacts/models/visss-mlx-v8/` — 14.7M params, d_model=384, 8 capas

**El salto más grande de precisión en todo el proyecto: +2.8pp sobre v7b.**

#### Cambios respecto a v7b

| Parámetro | v7b | v8 |
|---|---|---|
| d_model | 256 | **384** |
| n_layers | 6 | **8** |
| n_heads | 8 | 8 |
| ffn_dim | 1024 | **1536** |
| Params totales | 5M | **14.7M** |
| Scheduled Sampling | ✗ | **✓** |
| Best val accuracy | 88.6% | **91.4%** |

#### Scheduled Sampling — cómo funciona

El scheduled sampling cierra el gap train/inference gradualmente:

1. **Warmup (épocas 1–3):** `ss_prob = 0`. Puro teacher-forcing. El modelo establece predicciones base razonables.
2. **Post-warmup:** `ss_prob` crece linealmente de 0 → 0.5 sobre 37 épocas.
3. **Por época, 2 forward passes:**
   - Pass 1: teacher-forcing → predicciones del modelo
   - Pass 2: reemplazar `prev_limb[i]` con predicción propia con probabilidad `ss_prob` → calcular loss real
4. **Inferencia:** completamente autoregresiva (ss_prob efectivo = 1).

```python
# CRÍTICO: debe estar dentro de mx.compile
def step_ss(model, x, y, ss_prob, optimizer):
    logits_tf = model(x)                    # Pass 1: teacher-forced
    preds = mx.argmax(logits_tf, axis=-1)
    mask = mx.random.uniform(preds.shape) < ss_prob
    x_ss = replace_prev_limb(x, preds, mask)  # sustituir prev_limb
    def loss_fn(model):
        logits = model(x_ss)
        return cross_entropy(logits, y).mean()
    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

step_ss_compiled = mx.compile(step_ss, inputs=model.state)
```

#### Progresión de entrenamiento v8

| Época | Oracle Val Acc | AR Val Acc | AR Gap | ss_prob | Nota |
|---|---|---|---|---|---|
| 1 | 78.9% | 78.9% | 0.0pp | 0 | warmup |
| 3 | 80.2% | 80.1% | 0.1pp | 0 | warmup ends |
| **5** | **85.3%** | **85.3%** | 0.0pp | 0.014 | **PRIMER EPOCH SS → +3.3pp** |
| 9 | 88.6% | 88.5% | 0.1pp | 0.068 | v7b ceiling igualado |
| 10 | 89.1% | 89.1% | 0.0pp | 0.081 | record histórico |
| 13 | 90.0% | 90.0% | 0.0pp | 0.122 | supera barrera 90% |
| 19 | 91.1% | 91.1% | 0.0pp | 0.216 | supera 91% |
| **25** | **91.4%** | **91.3%** | **0.1pp** | 0.297 | **★ MEJOR EPOCH** |
| 37 | 91.1% | 91.1% | 0.0pp | 0.460 | early stop (patience 12/12) |

#### Hallazgos clave de v8

1. **AR gap ≈ 0.0pp durante todo el entrenamiento.** El modelo es completamente autoregresivo-compatible. No depende del oracle `prev_limb`. Esto significa que el próximo salto de accuracy no vendrá de mejorar el train/inference gap — vendrá de arquitectura o datos.

2. **El scheduled sampling causó el mayor salto individual:** +3.3pp en un solo epoch (época 5, primer epoch con SS). Incluso con ss_prob=0.014, el efecto regularizador es masivo.

3. **Convergencia 3× más rápida:** v8 iguala el mejor resultado de v7b (88.6%) en 9 épocas vs 30 de v7b.

4. **Bug crítico de MLX:** llamar `model()` fuera del step compilado causa `IndexError: unordered_map::at`. El scheduled sampling interno debe estar completamente dentro de `mx.compile`.

5. **Plateau claro:** épocas 25–37 oscilan entre 91.0–91.4%. El cuello de botella ya no es el entrenamiento — es la arquitectura y/o los datos.

#### Configuración técnica final v8

```python
config = {
    "d_model": 384,
    "n_heads": 8,
    "n_layers": 8,
    "ffn_dim": 1536,
    "dropout": 0.1,
    "n_input_features": 22,   # 18 arrow + 4 prev_limb one-hot
    "n_classes": 3,           # L=0, R=1, E=2
    "causal_mask": True,
    "ss_warmup_epochs": 3,
    "ss_max_prob": 0.5,
    "total_epochs": 40,
    "augmentation": "both-mirror",
    "MAX_SEQ_LEN": 1024,
    "CHUNK_OVERLAP": 256,
}
```

#### Progresión general del proyecto

```
LightGBM      75.4%  ──────────────────────────────┐
v7b (5M)      88.6%  ──────────────────────────────────────────────┐
v8 (14.7M)    91.4%  ══════════════════════════════════════════════════ ★
              ↑         ↑                              ↑
           Baseline  Transformer +              SS + modelo
           GBDT      causal mask                grande
```

---

## PARTE 2 — Roadmap futuro (hacia 94–96%)

### Análisis del plateau actual

v8 se estabilizó entre 91.0–91.4% por 12 épocas antes del early stop. El AR gap = 0.0pp confirma que **el cuello de botella ya no es el training pipeline sino la capacidad arquitectónica y los datos**. Las próximas mejoras atacan tres ejes:

- **(A) Decodificación** — cómo el modelo genera predicciones finales
- **(B) Arquitectura** — qué puede representar el modelo
- **(C) Datos** — cuánto y qué tipos de datos ve

---

### FASE 0 — Diagnóstico (1 día, obligatorio)

> **Sin esto, cualquier cambio arquitectónico es a ciegas.**

- [ ] **Confusion matrix por subgrupos:** chart_level, panel_type (DL/UL/C/UR/DR), is_bracket, run_length, dt_from_prev (rápido vs lento).
- [ ] **Clasificar el 8.6% de errores:** ¿son brackets? ¿doublesteps forzados? ¿jacks? ¿crossovers? ¿transiciones de BPM? ¿inicios de stream?
- [ ] **Comparar contra fefemz:** ¿qué porcentaje del gap restante es disagreement genuino vs ruido del ground truth (ambigüedad real)?

El resultado de Fase 0 determina cuáles de las siguientes fases son prioritarias.

---

### FASE 1 — Wins baratos sin cambiar arquitectura (+0.5 a +1.5pp esperado)

Estos cambios solo tocan la inferencia. Sin re-entrenar.

#### 1.1 — Test-Time Augmentation (TTA)

> **Esfuerzo:** 1 hora | **Ganancia esperada:** +0.3–0.5pp

Predecir con chart original Y chart espejado, invertir labels del espejo, promediar logits. Gratis dado que el modelo ya entiende ambas orientaciones (both-mirror training).

```python
# Inferencia con TTA
logits_orig = model(x_orig, padding_mask)
logits_mirror = model(x_mirror, padding_mask)
logits_mirror_flipped = flip_LR_logits(logits_mirror)  # intercambiar dim L y R
logits_final = (logits_orig + logits_mirror_flipped) / 2
preds = argmax(logits_final)
```

#### 1.2 — Beam Search Decoding

> **Esfuerzo:** 1 día | **Ganancia esperada:** +0.3–0.7pp

El modelo actual usa greedy argmax (decodificación codiciosa). Beam search con k=4–8 mantiene las K mejores hipótesis parciales y elige la de mayor probabilidad conjunta. Con solo 3 clases el espacio es pequeño y manejable.

#### 1.3 — Seed Ensemble (3–5 modelos v8)

> **Esfuerzo:** 2–3 días de entrenamiento en paralelo | **Ganancia esperada:** +0.5–1pp

Entrenar 3–5 modelos v8 idénticos con seeds distintos (solo difiere la inicialización aleatoria). Promediar logits en inferencia. Es el **upper bound práctico** de la arquitectura actual.

---

### FASE 2 — Cambios arquitectónicos (+2 a +5pp esperado)

#### 2.1 — Linear-Chain CRF Head + Viterbi Decoding ⭐ Mayor impacto/esfuerzo

> **Esfuerzo:** 1–2 días | **Ganancia esperada:** +1–2pp

Reemplazar el `OutputHead` de 2 capas por un CRF que modele transiciones explícitas:

```
transformer output → emisión de logits → CRF → Viterbi → secuencia óptima
```

La matriz de transición `T[y_i, y_{i-1}]` aprende restricciones físicas de facto:
- Qué tan probable es L→L (doublestep, poco probable salvo jacks)
- L→R vs L→L vs L→E dado el timing entre flechas
- La diferencia entre streams (L-R-L-R) vs brackets (L+R simultáneo)

**Loss:** negative log-likelihood del path Viterbi. **Inferencia:** Viterbi.

En `piu_annotate/ml/mlx_architecture.py`, agregar clase `CRFHead` después del `OutputHead` existente.

#### 2.2 — Bloques Transformer Modernos (RoPE + RMSNorm + SwiGLU)

> **Esfuerzo:** 1 día | **Ganancia esperada:** +0.3–0.7pp

Tres upgrades drop-in que caracterizan arquitecturas modernas (Llama, Gemma, etc.):

| Cambio | Reemplaza | Beneficio |
|---|---|---|
| **RoPE** (Rotary PE) | Sinusoidal PE | Mejor extrapolación a secuencias largas; relaciones relativas entre posiciones |
| **RMSNorm** | LayerNorm | Más estable, ligeramente más rápido |
| **SwiGLU FFN** | Linear+GELU+Linear | +0.3–0.5pp típico en LLMs |

```python
# SwiGLU: gate(x) ⊗ activation(x) en lugar de activation(linear(x))
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, ffn_dim):
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
```

#### 2.3 — Encoder Bidireccional + Decoder Autoregresivo ⭐ Mayor upside

> **Esfuerzo:** 1–2 semanas | **Ganancia esperada:** +1.5–3pp

Este es el cambio arquitectónico más grande y de mayor potencial.

**El problema actual:** La causal mask es necesaria para evitar leakage de `prev_limb`, pero también le prohíbe al modelo ver el contexto futuro legítimo. En PIU, las próximas 1–3 flechas son cruciales para decidir el pie actual (setups de brackets, evitar doublesteps imposibles).

**La solución:**
- **Encoder bidireccional** sobre las **18 arrow features** (sin `prev_limb`) → atiende pasado + futuro sin leakage (no hay información de label aquí).
- **Decoder causal** pequeño que toma la salida del encoder + `prev_limb` → genera el label actual.

```
18 arrow features → Encoder bidireccional → contexto rico [pasado + futuro]
                                             ↓
                    prev_limb (causal) → Decoder → label_i
```

Esto separa "qué hay en el chart" (visión completa) de "qué decidí hacer hasta ahora" (causal). Analógo a encoder-decoder de traducción automática.

#### 2.4 — Conformer (Convolución + Atención por capa)

> **Esfuerzo:** 2–3 días | **Ganancia esperada:** +0.5–1pp

PIU tiene estructura local muy fuerte: la decisión de pie depende principalmente de las 4–8 flechas vecinas. Una convolución 1D por capa captura este sesgo inductivo que la atención global no tiene naturalmente.

```
h = attn(h) + conv1d(h)   # en lugar de solo attn(h)
```

---

### FASE 3 — Datos y Pre-entrenamiento (+2 a +5pp esperado)

#### 3.1 — Pseudo-labeling sobre charts no anotados

> **Esfuerzo:** 1 semana | **Ganancia esperada:** +1–3pp (depende del volumen de corpus)

Si existe un corpus grande de charts sin anotación humana:

1. Usar el ensemble de v8 (Fase 1.3) para etiquetar todos los charts no anotados.
2. Filtrar por confianza (margin entre top-1 y top-2 > umbral, e.g., 0.9).
3. Entrenar v9 con datos originales + pseudo-labels filtrados.

Esta es potencialmente la **palanca más grande del proyecto** si el corpus no anotado es grande.

#### 3.2 — Pre-entrenamiento self-supervised (BERT-style)

> **Esfuerzo:** 2–4 semanas | **Ganancia esperada:** +1–2pp

Tarea: **Masked Arrow Modeling**. Enmascarar 15% de las flechas y predecir sus features (col_idx bucketizado, beat_time cuantizado, panel_type, etc.). Pre-entrenar en TODO el corpus (anotado + no anotado). Luego fine-tuning supervisado en la tarea de limb.

El pre-entrenamiento aprende representaciones de "qué tipo de flecha es" antes de saber "qué pie la toca".

#### 3.3 — Multi-task Auxiliary Heads

> **Esfuerzo:** 2–3 días | **Ganancia esperada:** +0.3–0.7pp

Agregar cabezas auxiliares que comparten los pesos del transformer pero predicen señales adicionales:

- **Predecir `is_bracket`** (¿es esta flecha parte de un corchete?) — loss peso 0.2
- **Predecir `is_doublestep_candidate`** — loss peso 0.1
- **Predecir `foot_dist` cuantizado** (¿qué tan lejos están los pies?) — loss peso 0.1

Gradiente más rico desde los mismos datos → mejor generalización en casos difíciles.

#### 3.4 — Transfer desde Doubles → Singles

> **Esfuerzo:** 1–2 semanas | **Ganancia esperada:** +0.5–1pp

Pre-entrenar en charts de doubles (10 paneles, más datos), luego fine-tuning en singles (5 paneles). El modelo aprende biomecánica de pies con más variedad.

---

## Tabla de priorización general

| Fase | Acción | Ganancia esperada | Esfuerzo | Riesgo | Orden |
|---|---|---|---|---|---|
| **0** | Error analysis completa | 0pp directos (dirige todo) | 1 día | Ninguno | **1° OBLIGATORIO** |
| **1.1** | TTA (test-time augmentation) | +0.3–0.5pp | 1 hora | Muy bajo | **2°** |
| **1.3** | Seed ensemble (3–5 modelos) | +0.5–1pp | 2–3 días | Muy bajo | **3°** |
| **1.2** | Beam search decoding | +0.3–0.7pp | 1 día | Bajo | **4°** |
| **2.1** | CRF head + Viterbi | +1–2pp | 1–2 días | Bajo | **5°** |
| **2.2** | RoPE + RMSNorm + SwiGLU | +0.3–0.7pp | 1 día | Bajo | **6°** |
| **2.4** | Conformer (conv + attn) | +0.5–1pp | 2–3 días | Medio | **7°** |
| **3.3** | Multi-task auxiliary heads | +0.3–0.7pp | 2–3 días | Bajo | **8°** |
| **2.3** | Encoder-decoder split | +1.5–3pp | 1–2 semanas | Medio | **9°** |
| **3.1** | Pseudo-labeling | +1–3pp | 1 semana | Medio | **10°** |
| **3.2** | Pre-entrenamiento BERT-style | +1–2pp | 2–4 semanas | Alto | **11°** |

---

## Proyección de accuracy objetivo

```
v8 actual:      91.4%  ████████████████████████████████████████░░░░░░░
+ TTA + Ensemble: ~92.5%  +1.1pp fáciles
+ CRF + Beam:   ~93.5%  +1.0pp sin re-entrenar arquitectura
+ Encoder-Decoder: ~95%  +1.5pp con nuevo diseño
+ Pseudo-labeling: ~96%  +1pp con más datos
────────────────────────────────────────────────────────────────
Target fefemz:  ~95–97%  (estimado de anotación humana experta)
```

**Objetivo realista en 6–8 semanas siguiendo el orden del plan:** **94–96%**

---

## Archivos clave del proyecto

| Archivo | Propósito |
|---|---|
| `piu_annotate/ml/mlx_architecture.py` | Arquitectura `LimbSequenceTransformer` (causal mask, OutputHead) |
| `cli/limbuse/train_mlx.py` | Loop de entrenamiento con step/step_ss compilados, AR eval |
| `cli/limbuse/cache_chunks.py` | Featurización de CSVs → .npz cacheados |
| `cli/limbuse/predict_limbs.py` | Inferencia sobre charts nuevos |
| `artifacts/models/visss-mlx-v8/` | Mejor modelo actual (91.4%) |
| `artifacts/cache/mlx-singles/` | Datos cacheados en .npz |

---

## Decisiones de diseño permanentes (no cambiar)

- **Causal mask:** Crítica. Sin ella hay data leakage por `prev_limb`. No se puede quitar a menos que se separe `prev_limb` del input (Fase 2.3).
- **Both-mirror augmentation:** Esencial para simetría L/R. Siempre mantener.
- **step_ss dentro de mx.compile:** Obligatorio. Llamar `model()` fuera del step compilado rompe MLX con `IndexError: unordered_map::at`.
- **mx.stop_gradient en x_ss:** Necesario dentro de step_ss para evitar que gradientes fluyan a través del argmax discreto.
- **Warmup de N épocas antes de SS:** Sin warmup, el modelo recibe predicciones caóticas como contexto `prev_limb` desde el inicio → entrenamiento inestable.
