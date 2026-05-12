from __future__ import annotations
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
    """ Use Tactician and PatternReasoner to predict limb annotations for `cs`
    """
    fcs = featurizers.ChartStructFeaturizer(cs)
    reasoner = PatternReasoner(cs, verbose = verbose)
    tactics = Tactician(cs, fcs, model_suite, verbose = verbose)

    is_transformer = getattr(model_suite, 'is_transformer', None)
    if is_transformer is None:
        is_transformer = getattr(model_suite, 'model_type', 'lightgbm') in ('mlx', 'torch')

    if is_transformer:
        # Transformer path: trust the model's sequence awareness, skip the
        # heavy reasoner/beam-search loop used for LGBM.
        pred_limbs = tactics.initial_predict()
    else:
        # Legacy path for LightGBM
        init_pred_limbs, abstained_lr_patterns = reasoner.propose_limbs()
        pred_limbs = tactics.initial_predict(init_pred_limbs, abstained_lr_patterns)
        
        score_to_limbs = dict()
        score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
        
        pred_limbs = tactics.enforce_arrow_after_hold_release(pred_limbs)
        score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
        
        pred_limbs = tactics.flip_labels_by_score(pred_limbs)
        score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
        
        pred_limbs = tactics.flip_jack_sections(pred_limbs)
        score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
        
        beam_results = tactics.beam_search(pred_limbs, width = 10, n_iter = 1)
        for pl in beam_results:
            score_to_limbs[tactics.score(pl)] = pl.copy()
            
        pred_limbs = tactics.fix_double_doublestep(beam_results[0])
        score_to_limbs[tactics.score(pred_limbs)] = pred_limbs.copy()
        
        pred_limbs = score_to_limbs[max(score_to_limbs)]

    # Final physical constraints (applied to both, but strictly enforced for MLX)
    pred_limbs = tactics.enforce_arrow_after_hold_release(pred_limbs)
    pred_limbs = tactics.detect_impossible_multihit(pred_limbs)
    pred_limbs = tactics.detect_impossible_lines_with_holds(pred_limbs)

    # Hard blacklist: deterministic fix for any remaining impossible same-foot brackets
    x_raw = fcs.get_raw_features()
    pred_limbs = fix_impossible_predictions(pred_limbs, x_raw[:, 0], x_raw[:, 6])

    if cs.get_chart_level() <= 15:
        pred_limbs = tactics.remove_unforced_brackets(pred_limbs)

    if verbose:
        fcs.evaluate(pred_limbs, verbose = True)

    return cs, fcs, pred_limbs