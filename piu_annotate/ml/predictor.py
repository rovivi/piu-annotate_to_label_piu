from __future__ import annotations
"""Limb prediction pipeline.

This is the **full upstream pipeline** applied to whichever backend is loaded
into ``ModelSuite``. The historical ``mlx`` fast-path that skipped reasoner +
beam-search has been removed because the upstream tactics buy 3–8pp of
accuracy on top of any backend's raw prediction. The cost is small (one
PatternReasoner pass + a width-5 / n-iter-3 beam search) and the scoring
function uses log-probabilities directly, so the transformer's softmax over
{L, R, E} feeds in without modification — argmax forcing to {L, R} happens
inside ``MLXModel.predict`` / ``TorchModel.predict`` already.

Pipeline (mirror of upstream piu_annotate/ml/predictor.py)
----------------------------------------------------------
    1. PatternReasoner.propose_limbs()    → high-confidence anchors + abstained
    2. tactics.initial_predict(anchors)   → model fills the rest, refine via arrowlimbs
    3. tactics.enforce_arrow_after_hold_release
    4. tactics.flip_labels_by_score       → per-token greedy flips
    5. tactics.flip_jack_sections         → uniform-foot enforcement on jacks
    6. tactics.beam_search(width=5, n_iter=3)   → 5+25+125 candidate refinement
    7. tactics.fix_double_doublestep      → physical pattern fix
    8. Keep argmax-score candidate from the score_to_limbs accumulator
    9. Final hard constraints: hold-release enforcement, impossible-multihit,
       impossible-lines-with-holds, blacklist fix
   10. Level ≤ 15 → remove_unforced_brackets

Tactician score for the transformer path: ``predict_arrowlimbs`` returns the
coarse softmax (or the refine cascade if a RefineModel is loaded);
``predict_matchnext`` / ``predict_matchprev`` are DummyMatchModel returning
log 0.5 — constant, so the beam search differentiates candidates purely by
the limb log-likelihood. When ``arrows_to_matchnext`` / ``arrows_to_matchprev``
get trained for the transformer backend (see
docs/next_gen_architecture.md §D), the score gains its 2nd and 3rd terms back.
"""
from loguru import logger
import numpy as np

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.tactics import Tactician
from piu_annotate.ml.models import ModelSuite
from piu_annotate.reasoning.reasoners import PatternReasoner
from piu_annotate.formats.notelines import fix_impossible_predictions


def predict(
    cs: ChartStruct,
    model_suite: ModelSuite,
    verbose: bool = False,
) -> tuple[ChartStruct, featurizers.ChartStructFeaturizer, np.ndarray]:
    fcs = featurizers.ChartStructFeaturizer(cs)
    reasoner = PatternReasoner(cs, verbose=verbose)
    tactics = Tactician(cs, fcs, model_suite, verbose=verbose)

    # ── Step 1: reasoner anchors ────────────────────────────────────────────
    init_pred_limbs, abstained_lr_patterns = reasoner.propose_limbs()

    # ── Step 2: ML initial predict (model + arrowlimbs refine if available) ─
    pred_limbs = tactics.initial_predict(init_pred_limbs, abstained_lr_patterns)
    score_to_limbs = {tactics.score(pred_limbs): pred_limbs.copy()}
    if verbose:
        logger.info(f'Score, initial pred: {tactics.score(pred_limbs):.3f}')

    # ── Step 3: hold-release coherence ──────────────────────────────────────
    pred_limbs = tactics.enforce_arrow_after_hold_release(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, arrow after hold release: {tactics.score(pred_limbs):.3f}')

    # ── Step 4: per-token greedy flips ──────────────────────────────────────
    pred_limbs = tactics.flip_labels_by_score(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip labels: {tactics.score(pred_limbs):.3f}')

    # ── Step 5: jack-section uniform foot ───────────────────────────────────
    pred_limbs = tactics.flip_jack_sections(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, flip jacks: {tactics.score(pred_limbs):.3f}')

    # ── Step 6: beam search (5 + 25 + 125 candidates) ───────────────────────
    # Uses the highest-scoring candidate so far as the seed.
    seed = score_to_limbs[max(score_to_limbs)]
    pred_limbs = tactics.beam_search(seed, width=5, n_iter=3)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, beam search: {tactics.score(pred_limbs):.3f}')

    # ── Step 7: double-doublestep correction ────────────────────────────────
    pred_limbs = tactics.fix_double_doublestep(pred_limbs)
    score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
    if verbose:
        logger.info(f'Score, fix double doublestep: {tactics.score(pred_limbs):.3f}')

    # ── Step 8: pick the best candidate seen ────────────────────────────────
    pred_limbs = score_to_limbs[max(score_to_limbs)]
    if verbose:
        logger.success(f'Best score found: {max(score_to_limbs):.3f}')

    # ── Step 9: final hard constraints ──────────────────────────────────────
    pred_limbs = tactics.enforce_arrow_after_hold_release(pred_limbs)
    pred_limbs = tactics.detect_impossible_multihit(pred_limbs)
    pred_limbs = tactics.detect_impossible_lines_with_holds(pred_limbs)

    # Temporal constraint: same foot twice in < 40ms on non-bracketable panels
    # is physically impossible at any realistic BPM → hard-alternate.
    pred_limbs = tactics.enforce_fast_note_alternation(pred_limbs)

    # Hard blacklist: deterministic fix for any remaining impossible same-foot
    # brackets. Specific to this fork (not in upstream); cheap insurance.
    x_raw = fcs.get_raw_features()
    pred_limbs = fix_impossible_predictions(pred_limbs, x_raw[:, 0], x_raw[:, 6])

    # ── Step 10: low-level unforced-bracket cleanup ────────────────────────
    if cs.get_chart_level() <= 15:
        pred_limbs = tactics.remove_unforced_brackets(pred_limbs)

    if verbose:
        fcs.evaluate(pred_limbs, verbose=True)

    return cs, fcs, pred_limbs
