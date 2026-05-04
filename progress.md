# Progress Report - PIU Limb Annotation

## Stage: Triple-Tap Stabilization & GBDT Scaling
**Status:** Completed
**Date:** 2026-05-04

### 1. What was done in this stage
- **Ambiguity Resolution:** Modified `relabel_with_ambiguous_e` logic to uniformly apply the `e` label to center panels in triple-taps.
- **Model Suite Retraining:** Retrained the LightGBM models with the full dataset, mapping `e` to `0` (Left) during training for stability.
- **Full Database Inference:**
    - Cleaned `artifacts/processed_db`.
    - Processed **637 songs** and **4,418 charts**.
    - Exported all results to JSON.
- **Production Deployment:** Synchronized 4,418 JSON files to the `piulatam` web server and pushed all code to GitHub.

### 2. What improved
- **Consistency:** Eliminated erratic limb assignment in triple-tap sequences (e.g., "Wicked Legend", "Brown Sky").
- **Accuracy:**
    - **Doubles:** Reached **96.5%**, exceeding previous stability benchmarks.
    - **Triple-taps:** Restored to **86.9%** (was previously dropping due to forced binary decisions).
- **Cleanliness:** The production environment is now 100% synchronized with the latest model output, removing duplicates from old runs.

### 3. What is missing
- **Singles Precision:** Accuracy is still at **75.4%**. The model fails to distinguish between bracket preferences in complex high-level charts.
- **Physical Validity:** Approximately 5-10% of charts still report "impossible line with holds" warnings.
- **Sequence Context:** The current model only looks at a small window of notes; it doesn't "understand" the physical momentum of the player.

### 4. Roadmap & Ideas for Improvement
**My Recommendation:**
The LightGBM architecture has hit a statistical ceiling. To reach >90% in Singles, we must transition from "Pattern Matching" (GBDT) to "Sequence Prediction" (Transformers).

1. **Phase 2 - Sequence Aware Modeling (Transformers):**
   - Implement an **MLX-based Transformer** to process charts as sequences.
   - This will allow the model to remember which foot is where across several measures, solving the bracket vs. jump ambiguity in Singles.

2. **Phase 3 - Tactician Refinement:**
   - Add a "Body Rotation Penalty" to the tactician to discourage unrealistic spins.
   - Refine the hold-note constraint solver to handle "paralelas" (3+ notes) more naturally.

3. **Phase 4 - Manual Correction Layer:**
   - Build the `check_hands` CLI to allow rapid manual auditing of the few charts that remain "impossible" after the MLX upgrade.
