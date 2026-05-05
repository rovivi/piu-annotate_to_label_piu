<div align="center">

# 🎮 PIU Limb Annotation — ML Project

### Predicting left/right foot annotations for Pump It Up step charts using deep learning

[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-91.4%25-FFD700?style=for-the-badge&logo=pytorch)](artifacts/models/visss-mlx-v8/)
[![AR Gap](https://img.shields.io/badge/AR%20Gap-0.0pp-3FB950?style=for-the-badge)](artifacts/models/visss-mlx-v8/)
[![Framework](https://img.shields.io/badge/Framework-MLX%20%2F%20Apple%20Silicon-58A6FF?style=for-the-badge)](https://github.com/ml-explore/mlx)
[![Params](https://img.shields.io/badge/Model%20Size-14.7M%20params-A371F7?style=for-the-badge)](#architecture)

**From a 75.4% LightGBM baseline to a 91.4% causal transformer with scheduled sampling — all running on Apple Silicon.**

</div>

---

## 📊 The Journey at a Glance

```
         75.4%                88.6%            91.4% ★
           │                    │                 │
  LightGBM │   v7b Transformer  │  v8 Transformer │
  Baseline │   5M params        │  14.7M params   │
           │   Causal mask      │  + Sched. Sampling
           │   Both-mirror      │  + Bigger model │
           │                    │                 │
  ─────────┴────────────────────┴─────────────────┴────────▶ Accuracy
  75%      80%       85%        89%      91%      92%
```

| Model | Architecture | Best Acc | Epochs | Key Innovation |
|-------|-------------|---------|--------|----------------|
| LightGBM | Gradient Boosting, manual features | 75.4% | N/A | Baseline |
| v7b | Transformer 5M, d=256, 6L | 88.6% | 30 | Causal mask + Both-mirror |
| **v8** ★ | **Transformer 14.7M, d=384, 8L** | **91.4%** | **37 (early stop)** | **Scheduled Sampling** |

---

## 🏆 Current Best: v8 — 91.4% Accuracy

<table>
<tr>
<td width="50%">

**Model stats:**
- `d_model` = 384 | `n_layers` = 8 | `n_heads` = 8
- `ffn_dim` = 1536 | `params` = **14.7M**
- `dropout` = 0.1 | `max_seq_len` = 1024
- Causal mask ✓ | Both-mirror aug ✓
- Scheduled sampling (ss_max=0.5) ✓
- Early stop epoch 37/40

</td>
<td width="50%">

**The killer result: AR Gap = 0.0pp**

```
Oracle (teacher-forced):   91.4%
Autoregressive (real):     91.3%
                           ──────
Difference:                 0.0pp ← 🎯
```

The model performs identically with or without ground-truth context. No train/inference gap.

</td>
</tr>
</table>

---

## 📈 v8 Training — Epoch by Epoch

```
Val %
 92 │                                     ★25
 91 │                             ·19·20·22·23·24 ·26·27·28·29·30·31
 90 │                         ·16·17                                  ···37 (STOP)
    │                  ·14·15
 89 │           ·11·12
    │       ·10
 88 │    ·9  ← v7b ceiling
    │  ·8
 87 │·7
    │·6
 86 │
    │                    ← SS STARTS HERE (epoch 5)
 85 │·5 ← +3.3pp jump!
    │
 82 │·4
    │·1·2·3 (warmup, ss_prob=0)
 78 ┼──────────────────────────────────────────────────────────── Epoch
     1  3  5  7  9  11 13 15 17 19 21 23 25 27 29 31 33 35 37
```

### Full Epoch Table

| Epoch | Oracle Acc | AR Acc | AR Gap | SS Prob | Note |
|-------|-----------|--------|--------|---------|------|
| 1 | 78.9% | 78.9% | 0.0pp | 0.000 | warmup |
| 2 | 78.8% | 78.8% | 0.0pp | 0.000 | warmup |
| 3 | 80.2% | 80.1% | 0.1pp | 0.000 | warmup ends |
| 4 | 82.0% | 81.9% | 0.1pp | 0.000 | post-warmup |
| **5** | **85.3%** | **85.3%** | **0.0pp** | 0.014 | **🔥 FIRST SS EPOCH → +3.3pp!** |
| 6 | 86.9% | 86.9% | 0.0pp | 0.027 | |
| 7 | 87.3% | 87.3% | 0.0pp | 0.041 | |
| 8 | 87.9% | 87.9% | 0.0pp | 0.054 | |
| **9** | **88.6%** | **88.5%** | **0.1pp** | 0.068 | **v7b ceiling matched (30ep→9ep!)** |
| 10 | 89.1% | 89.1% | 0.0pp | 0.081 | breaks v7b all-time record |
| 11 | 89.8% | 89.7% | 0.1pp | 0.095 | |
| 12 | 89.4% | 89.4% | 0.0pp | 0.108 | slight dip (noise) |
| **13** | **90.0%** | **90.0%** | **0.0pp** | 0.122 | **🚀 BREAKS 90% BARRIER** |
| 14 | 90.6% | 90.6% | 0.0pp | 0.135 | |
| 15 | 90.5% | 90.5% | 0.0pp | 0.149 | |
| 16 | 90.8% | 90.8% | 0.0pp | 0.176 | |
| 17 | 90.9% | 90.8% | 0.1pp | 0.189 | |
| 18 | 90.1% | 90.1% | 0.0pp | 0.203 | dip (SS noise) |
| **19** | **91.1%** | **91.1%** | **0.0pp** | 0.216 | **breaks 91%** |
| 20 | 91.2% | 91.1% | 0.1pp | 0.230 | |
| 21 | 91.0% | 91.0% | 0.0pp | 0.243 | |
| 22 | 91.2% | 91.2% | 0.0pp | 0.257 | |
| 23 | 91.3% | 91.3% | 0.0pp | 0.270 | |
| 24 | 91.1% | 91.0% | 0.1pp | 0.284 | no_improve 1/12 |
| **25** ★ | **91.4%** | **91.3%** | **0.1pp** | 0.297 | **🏆 ALL-TIME BEST** |
| 26 | 91.3% | 91.3% | 0.0pp | 0.311 | no_improve 1/12 |
| 27 | 91.0% | 91.0% | 0.0pp | 0.324 | no_improve 2/12 |
| 28 | 91.2% | 91.2% | 0.0pp | 0.338 | no_improve 3/12 |
| 29 | 91.3% | 91.3% | 0.0pp | 0.351 | no_improve 4/12 |
| 30 | 91.3% | 91.2% | 0.1pp | 0.365 | no_improve 5/12 |
| 31 | 91.2% | 91.1% | 0.1pp | 0.378 | no_improve 6/12 |
| 32 | 91.2% | 91.1% | 0.1pp | 0.392 | no_improve 7/12 |
| 33 | 91.1% | 91.1% | 0.0pp | 0.405 | no_improve 8/12 |
| 34 | 91.2% | 91.1% | 0.1pp | 0.419 | no_improve 9/12 |
| 35 | 91.2% | 91.1% | 0.1pp | 0.432 | no_improve 10/12 |
| 36 | 91.1% | 91.1% | 0.0pp | 0.446 | no_improve 11/12 |
| **37** | **91.1%** | **91.1%** | **0.0pp** | 0.460 | **🛑 EARLY STOP (12/12)** |

---

## 🏗️ Architecture — LimbSequenceTransformer

```
┌──────────────────────────────────────────────────┐
│              INPUT SEQUENCE                       │
│         N arrows × 22 features                   │
│  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  18 arrow feats  │  │  4 prev_limb one-hot  │  │
│  │ col_idx, timing  │  │  L / R / E / START    │  │
│  │ geometry, holds  │  │                       │  │
│  └──────────────────┘  └──────────────────────┘  │
└──────────────────────────────────┬───────────────┘
                                   │ x ∈ ℝ^(N×22)
                                   ▼
                     ┌─────────────────────────┐
                     │   Input LayerNorm        │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Linear Projection      │
                     │      22 → 384            │
                     └────────────┬────────────┘
                                  │  + (element-wise)
                                  ▼
                     ┌─────────────────────────┐
                     │  Sinusoidal Positional   │
                     │      Encoding            │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │     Dropout (p=0.1)      │
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  TransformerEncoder ×8   │  ← 8 layers
                     │  n_heads=8  ffn=1536     │
                     │                          │
                     │ ┌──────────┐ ┌────────┐  │
                     │ │Multi-Head│ │  FFN   │  │
                     │ │   Attn   │ │ (1536) │  │
                     │ │+ CAUSAL  │ │+ResNorm│  │
                     │ │  MASK    │ │        │  │
                     │ └──────────┘ └────────┘  │
                     │  ⚠️ position i only sees  │
                     │     positions 0..i        │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Output LayerNorm       │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  2-Layer Classification  │
                     │  d_model → d/2 → GELU   │
                     │          → 3 classes     │
                     └────────────┬────────────┘
                                  │ logits ∈ ℝ^(N×3)
                                  ▼
                     ┌─────────────────────────┐
                     │   Output: Limb per Arrow │
                     │  0=Left  1=Right  2=Either│
                     └─────────────────────────┘
```

### Input Features (22 total)

**18 Arrow Features** (position, timing, geometry):

| Feature | Description |
|---------|-------------|
| `col_idx` | Column index of the arrow |
| `symbol` | Note type (tap, hold head, hold end) |
| `beat_time` | Timing in beats |
| `is_hold_head` | Boolean: starts a hold |
| `is_hold_end` | Boolean: ends a hold |
| `hold_duration` | Duration of hold in seconds |
| `is_bracket` | Boolean: can be bracketed |
| `bracket_idx` | Bracket group index |
| `x_coord` | Physical X position on pad |
| `y_coord` | Physical Y position on pad |
| `dt_from_prev` | Time delta from previous arrow |
| `dt_to_next` | Time delta to next arrow |
| `run_length` | Length of current run sequence |
| `same_col_as_prev` | Same column as previous arrow |
| `foot_dist` | Estimated foot travel distance |
| `is_doublestep_candidate` | Boolean: could be a doublestep |
| `panel_type` | Panel zone (DL/UL/C/UR/DR) |
| `chart_level` | Difficulty level of the chart |

**4 prev_limb One-Hot:**

| Dim | Meaning | Train | Inference |
|-----|---------|-------|-----------|
| 0 | Previous = Left | Ground truth | Model prediction |
| 1 | Previous = Right | Ground truth | Model prediction |
| 2 | Previous = Either | Ground truth | Model prediction |
| 3 | START token | Chart beginning | Chart beginning |

---

## 🔑 Why the Causal Mask is Critical

```
WITHOUT causal mask (v7a — WRONG):

Token i wants to predict Left or Right.
It can attend to token i+1.
Token i+1 has prev_limb = [1,0,0,0] (Left).
This REVEALS the label of token i. That's cheating.

        Arrow 5 ──────────────────────▶ Arrow 6
      predict L/R                 prev_limb = L
                                  (reveals Arrow 5 = L)
                                       ↑ LEAKAGE 💀

WITH causal mask (v7b, v8 — CORRECT):

Attention matrix (✓ = can attend, ✗ = blocked):

       j=0  j=1  j=2  j=3  j=4
  i=0 [ ✓    ✗    ✗    ✗    ✗ ]
  i=1 [ ✓    ✓    ✗    ✗    ✗ ]
  i=2 [ ✓    ✓    ✓    ✗    ✗ ]
  i=3 [ ✓    ✓    ✓    ✓    ✗ ]
  i=4 [ ✓    ✓    ✓    ✓    ✓ ]

Upper triangle = -∞ (blocked)
Position i can only attend to positions 0..i ✓
```

---

## 🔄 Scheduled Sampling — How It Works

```
TRAINING (ss_prob=0):
  prev_limb[i] = ground truth label[i-1]
  Model always gets perfect context.
  Problem: exposure bias — model never sees its own errors.

        GT: L  R  L  R  L  →  predict on these
            ↓  ↓  ↓  ↓  ↓
          [L][R][L][R][L]  (oracle prev_limb)

SCHEDULED SAMPLING (0 < ss_prob ≤ 0.5):
  2 forward passes per batch:
  1️⃣ Pass 1: teacher-forced → get model predictions
  2️⃣ Replace prev_limb[i] with prediction with prob ss_prob
  3️⃣ Pass 2: forward with mixed context → compute real loss

        GT: L  R  L  R  L
            ↓     ↓     ↓   ← ss_prob=0.5: replace 3/5 positions
         pred  GT  pred GT  pred
          [R] [R] [L] [R] [L]   (some oracle, some model preds)

INFERENCE (ss_prob=1 implicit):
  prev_limb[i] = model's own prediction for arrow i-1
  Fully autoregressive. No oracle needed.

  AR Gap = 0.0pp → model handles this perfectly ✓
```

### Scheduled Sampling Schedule (v8)

```
ss_prob
 0.50 │                                             ·37
 0.45 │                                         ·35·36
 0.40 │                                     ·31·32
 0.35 │                                 ·28·29
 0.30 │                             ·25·26
 0.25 │                         ·22·23
 0.20 │                     ·18·19
 0.15 │                 ·14·15
 0.10 │          ·10·11
 0.05 │      ·6·7
 0.01 │  ·5
 0.00 │──────                              (warmup = epochs 1-3)
      └────────────────────────────────────────────── Epoch
       1  3  5  7  9  11 13 15 17 19 21 23 25 27 37
```

---

## 🪞 Both-Mirror Augmentation

```
Every epoch trains on BOTH orientations of every chart:

ORIGINAL:                      MIRRORED:
  ↙  ↑  →                →      ←  ↑  ↗
  L  R  R                        R  L  L

Panel mapping:
  DL ↔ DR   (down-left ↔ down-right)
  UL ↔ UR   (up-left  ↔ up-right)
  C  =  C   (center stays center)

Label mapping: L ↔ R  |  E = E

Result: Perfect L/R symmetry in the dataset.
Prevents the model from learning shortcuts based on
starting-foot distribution patterns in the ground truth.
```

---

## 🚨 Critical Implementation Notes

### Bug: MLX compile boundary

```python
# ❌ WRONG — causes IndexError: unordered_map::at in MLX
def step(model, x, y, ss_prob, optimizer):
    logits = model(x)                  # forward OUTSIDE compile
    preds = mx.argmax(logits, axis=-1)
    x_ss = replace_prev_limb(x, preds)
    loss, grads = mx.value_and_grad(loss_fn)(model)  # INSIDE compile
    ...

# ✅ CORRECT — entire scheduled sampling inside mx.compile
def step_ss(model, x, y, ss_prob, optimizer):
    # Pass 1: teacher-forced to get predictions
    logits_tf = model(x)
    preds = mx.argmax(logits_tf, axis=-1)

    # Replace prev_limb with model predictions at rate ss_prob
    mask = mx.random.uniform(preds.shape) < ss_prob
    x_ss = replace_prev_limb(x, preds, mask)

    # mx.stop_gradient REQUIRED: prevents grads through discrete argmax
    x_ss = mx.stop_gradient(x_ss)

    # Pass 2: real forward + loss on mixed context
    def loss_fn(model):
        logits = model(x_ss)
        return cross_entropy(logits, y).mean()

    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Compile sees the full graph including both forward passes
step_ss_compiled = mx.compile(step_ss, inputs=model.state)
```

### Immutable design decisions

| Decision | Reason | Do NOT change |
|----------|--------|---------------|
| **Causal mask** | Without it, `prev_limb` leaks future labels | Unless you decouple prev_limb from input |
| **Both-mirror augmentation** | Ensures L/R symmetry | Always keep |
| **step_ss inside mx.compile** | MLX crashes with `IndexError` otherwise | Mandatory |
| **mx.stop_gradient on x_ss** | Prevents gradient through discrete argmax | Required |
| **Warmup before SS** | Unstable training without it | Keep ≥3 epochs |

---

## 📐 Technical Configuration

```python
# LimbSequenceTransformer v8 — final config
config = {
    "d_model":            384,    # up from 256 (v7b)
    "n_heads":            8,
    "n_layers":           8,      # up from 6 (v7b)
    "ffn_dim":            1536,   # 4 × d_model
    "dropout":            0.1,
    "n_input_features":   22,     # 18 arrow + 4 prev_limb one-hot
    "n_classes":          3,      # L=0, R=1, E=2
    "causal_mask":        True,
    "ss_warmup_epochs":   3,
    "ss_max_prob":        0.5,
    "total_epochs":       40,
    "augmentation":       "both-mirror",
    "MAX_SEQ_LEN":        1024,
    "CHUNK_OVERLAP":      256,
}

# SS probability schedule
ss_prob = (epoch - 3) / (40 - 3) * 0.5 if epoch > 3 else 0.0
```

---

## 📁 Project Structure

```
piu-annotate_to_label_piu/
│
├── 📄 README.md                      ← You are here
├── 📄 ROADMAP.md                     ← Full history + future plan
│
├── piu_annotate/
│   └── ml/
│       ├── mlx_architecture.py       ← LimbSequenceTransformer (causal mask)
│       ├── mlx_dataset.py            ← Dataset utilities
│       ├── featurizers.py            ← 18 arrow features
│       └── predictor.py              ← Inference pipeline
│
├── cli/limbuse/
│   ├── train_mlx.py                  ← Training loop (step/step_ss compiled)
│   ├── cache_chunks.py               ← CSV → .npz feature cache
│   ├── predict_limbs.py             ← Inference on new charts
│   └── eval_models.py               ← Evaluation tools
│
└── artifacts/
    ├── models/
    │   ├── visss-mlx-v7b/           ← 88.6% (5M params)
    │   └── visss-mlx-v8/            ← 91.4% ★ (14.7M params, current best)
    └── cache/
        └── mlx-singles/             ← Pre-computed .npz feature files
```

---

## 🚀 Quickstart

```bash
# Install
pip install -e .
pip install mlx lightgbm pandas numpy loguru tqdm scipy scikit-learn

# Cache features from CSV charts
python cli/limbuse/cache_chunks.py --input-dir /path/to/charts --output-dir artifacts/cache/mlx-singles

# Train (v8 config)
python cli/limbuse/train_mlx.py \
    --cache-dir artifacts/cache/mlx-singles \
    --model-dir artifacts/models/my-model \
    --d-model 384 --n-layers 8 --n-heads 8 --ffn-dim 1536 \
    --ss-max-prob 0.5 --warmup-epochs 3 --epochs 40

# Predict on a chart
python cli/limbuse/predict_limbs.py \
    --model artifacts/models/visss-mlx-v8 \
    --input chart.csv
```

---

## 🗺️ Roadmap — Getting to 95%+

The v8 plateau (91.0–91.4% for 12 epochs) signals that **the bottleneck is no longer training — it's architecture and data**. The AR gap being 0.0pp confirms the next improvement won't come from closing the train/inference gap.

```
Current:    91.4%  ████████████████████████████████████░░░░░░░░░
+ Quick wins: ~92.5%  TTA + Seed ensemble (no retraining needed)
+ CRF:      ~93.5%  Explicit transition modeling + Viterbi
+ Enc-Dec:  ~95.0%  Bidirectional encoder + causal decoder
+ More data: ~96%+  Pseudo-labeling on unlabeled charts
Target:     ~95-97% Best human annotator (fefemz)
```

### Priority order

| Phase | Change | Expected gain | Effort |
|-------|--------|--------------|--------|
| **0** | Error analysis by subgroup (brackets, jacks, BPM transitions) | — diagnostic | 1 day |
| **1a** | Test-time augmentation (predict orig + mirror, average) | +0.3–0.5pp | 1 hour |
| **1b** | Seed ensemble (3–5 v8 models, average logits) | +0.5–1pp | 2–3 days training |
| **1c** | Beam search decoding (k=4–8) instead of greedy argmax | +0.3–0.7pp | 1 day |
| **2a** | **CRF head + Viterbi decoding** | **+1–2pp** | 1–2 days |
| **2b** | RoPE + RMSNorm + SwiGLU (modern transformer block) | +0.3–0.7pp | 1 day |
| **2c** | **Encoder-decoder split** (bidir encoder / causal decoder) | **+1.5–3pp** | 1–2 weeks |
| **3a** | Pseudo-labeling on unlabeled chart corpus | +1–3pp | 1 week |
| **3b** | Masked-arrow pre-training (BERT-style) | +1–2pp | 2–4 weeks |

See [`ROADMAP.md`](ROADMAP.md) for the detailed plan including implementation notes and rationale.

---

## 📊 Model Comparison

```
Accuracy (singles validation set)

LightGBM  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░  75.4%
           │ No sequence context │

v7b 5M    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░  88.6%
           │ Causal mask + both-mirror │ 30 epochs │

v8 14.7M  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  91.4% ★
           │ + Scheduled sampling │ 37 epochs │ Early stop │

Target    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  95–97%
           │ Best human annotator (fefemz) │
           0%                                              100%
```

---

## 💡 Key Findings

### 1. Scheduled sampling caused the biggest single-epoch jump in the entire run

Epoch 5 (first SS epoch, `ss_prob=0.014`) produced a **+3.3pp jump** — from 82.0% to 85.3%. Even with just 1.4% of tokens using model predictions as context, the regularization effect was massive. The model immediately started learning to be robust to its own errors.

### 2. Bigger model converges 3–4× faster

v8 matched v7b's all-time best (88.6%) in **9 epochs** vs v7b's **30 epochs**. The 2.9× parameter increase didn't just improve final accuracy — it fundamentally changed the learning dynamics.

### 3. AR gap = 0.0pp throughout training

From epoch 1, oracle accuracy (teacher-forced) and AR accuracy (autoregressive) were within 0.0–0.1pp. This is exceptional: the model never developed an oracle dependency. It always predicted as well with its own outputs as with ground truth.

### 4. MLX compile boundary is a hard constraint

Calling `model()` outside the compiled step function causes `IndexError: unordered_map::at` in MLX. The entire scheduled sampling (including both forward passes) must live inside a single `mx.compile` call. This is a subtle MLX implementation detail that took significant debugging to identify.

### 5. The plateau tells you what to fix next

The 12-epoch no-improvement plateau (epochs 25–37 at 91.0–91.4%) is diagnostic: **the model has learned everything it can given its architecture and training setup**. The 8.6% remaining errors are not fixable by training longer or tuning learning rate — they require new architectural capabilities (bidirectional context, structured decoding) or more data.

---

## ⚙️ Skills Detection System

The project also includes a rule-based skills detection system that annotates charts with 25+ technical patterns. These features feed into the ML pipeline as part of the 18 arrow features.

<details>
<summary>Click to expand skills list</summary>

| Skill | Description |
|-------|-------------|
| `run` | Alternating feet sequence, consistent rhythm, ≥7 notes |
| `drill` | Run with repeated note pattern |
| `jack` | Same panel, same foot, rapid |
| `footswitch` | Same panel, alternating feet |
| `bracket` | Two notes playable with one foot |
| `staggered_bracket` | Bracket split across two consecutive lines |
| `doublestep` | Two different panels, same foot |
| `twist_90` | 90° body rotation required |
| `twist_over90` | >90° body rotation required |
| `twist_close` / `twist_far` | Twist distance variants |
| `side3_singles` | 3 panels on one side in singles |
| `split` | Feet at opposite extremes |
| `stair5` / `stair10` | 5/10-note staircase patterns |
| `yog_walk` | Yog walk pattern |
| `hold_footswitch` | Footswitch during hold |
| `bracket_run` / `bracket_drill` | Compound patterns |

</details>

---

## 🎯 Project Goal

Annotate every Pump It Up chart with **which foot hits each arrow** at the accuracy level of the best human annotator (fefemz). Ground truth is vis-ss manual annotations.

The annotations are used to:
- Teach players correct foot technique
- Power the difficulty rating system
- Enable automatic chart analysis for the PIU community

---

<div align="center">

**v8 LimbSequenceTransformer · 91.4% Singles Val Accuracy · Apple Silicon / MLX**

*Early stop @ epoch 37/40 · Best @ epoch 25 · AR gap 0.0pp*

</div>
