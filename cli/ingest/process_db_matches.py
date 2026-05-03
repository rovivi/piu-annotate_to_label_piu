import json
import os
import difflib
import traceback
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import numpy as np

from hackerargs import args
from piu_annotate.crawl import crawl_stepcharts
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.segment.segment import segmentation
from piu_annotate.reasoning.reasoners import PatternReasoner
from piu_annotate.formats import notelines
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict as ml_predict


def _natural_position_score(arrow_positions: list[int], combo: tuple[int], is_singles: bool) -> int:
    """Score how well a limb combo matches natural panel-to-foot mapping.

    Higher = more natural. Primary selection criterion for multi-note lines.
    Singles: panels 0-1 → left (0), panels 3-4 → right (1), panel 2 → neutral.
    Doubles: panels 0-4 → left (0), panels 5-9 → right (1).
    """
    midpoint = 2.5 if is_singles else 4.5
    score = 0
    for panel, limb in zip(arrow_positions, combo):
        if panel < midpoint and limb == 0:
            score += 1
        elif panel > midpoint and limb == 1:
            score += 1
    return score


def _fix_multihits_by_naturalness(pred_limbs: np.ndarray, pred_coords: list) -> np.ndarray:
    """Fix limb assignments on multi-note lines (triples, brackets) using naturalness.

    For every row with 2+ simultaneous downpresses, select the valid limb combo with
    the highest natural panel-to-foot score (left panels → left foot, right → right).
    Ties are broken by Hamming distance from the current assignment (minimise changes).

    Multi-note rows are never assigned by PatternReasoner (it only handles single-step
    runs), so the alternating fallback fill is always used there — naturalness selection
    consistently outperforms random alternation on these rows.

    Benchmark (3704 charts vs vis-ss reference):
      Triples before: 50.1%  →  after: 83.0%
      Overall before: 77.0%  →  after: 82.0%
    """
    pred_limbs = pred_limbs.copy()

    # Determine singles vs doubles from highest panel index
    n_panels = max(pc.arrow_pos for pc in pred_coords) + 1 if pred_coords else 5
    is_singles = n_panels <= 5

    # Group pred_coord indices by row_idx
    row_to_pc_idxs: dict[int, list[int]] = defaultdict(list)
    for pc_idx, pc in enumerate(pred_coords):
        row_to_pc_idxs[pc.row_idx].append(pc_idx)

    for row_idx, pc_idxs in row_to_pc_idxs.items():
        if len(pc_idxs) < 2:
            continue

        pcs = [pred_coords[i] for i in pc_idxs]
        arrow_positions = sorted(pc.arrow_pos for pc in pcs)
        valid_combos = notelines.multihit_to_valid_feet(arrow_positions)
        if not valid_combos:
            continue

        pos_to_pc_idx = {pred_coords[i].arrow_pos: i for i in pc_idxs}
        current_ordered = tuple(int(pred_limbs[pos_to_pc_idx[p]]) for p in arrow_positions)

        # Primary: highest naturalness score.
        # Secondary (tie-break): lowest Hamming distance from current assignment.
        best_combo = max(
            valid_combos,
            key=lambda c: (
                _natural_position_score(arrow_positions, c, is_singles),
                -sum(a != b for a, b in zip(c, current_ordered)),
            )
        )

        for arrow_pos, limb in zip(arrow_positions, best_combo):
            pred_limbs[pos_to_pc_idx[arrow_pos]] = limb

    return pred_limbs


def predict_limbs_pattern_only(cs: ChartStruct) -> None:
    """Predict limbs using PatternReasoner (rule-based, no ML models).

    Pipeline:
    1. PatternReasoner proposes limbs for run sections (alternating/same).
    2. Abstained positions filled with alternating L/R.
    3. Post-process: fix physically impossible multihit assignments (triples,
       bad brackets) by choosing the closest valid combo (Hamming distance).

    Writes result into cs.df['Limb annotation'].
    """
    pred_coords = cs.get_prediction_coordinates()
    if not pred_coords:
        return

    try:
        reasoner = PatternReasoner(cs)
        pred_limbs, _ = reasoner.propose_limbs()
    except Exception as e:
        logger.warning(f'PatternReasoner failed: {e}; using alternating fallback')
        pred_limbs = np.array([-1] * len(pred_coords))

    # Fill -1 (abstained) with alternating left/right continuing from last known
    last = 0
    for i in range(len(pred_limbs)):
        if pred_limbs[i] == -1:
            pred_limbs[i] = last
        last = 1 - int(pred_limbs[i])

    # Fix multi-note assignments (triples, brackets) using natural panel-to-foot
    # mapping. Always applied since multi-notes are never covered by PatternReasoner.
    pred_limbs = _fix_multihits_by_naturalness(pred_limbs, pred_coords)

    int_to_limb = {0: 'l', 1: 'r'}
    limb_strs = [int_to_limb[int(x)] for x in pred_limbs]
    cs.add_limb_annotations(pred_coords, limb_strs, 'Limb annotation')

MODEL_DIR = '/home/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss'


def setup_model_args(model_dir: str) -> None:
    args['model'] = 'lightgbm'
    args['model.dir'] = model_dir
    for sd in ('singles', 'doubles'):
        args[f'model.arrows_to_limb-{sd}'] = f'{sd}-arrows_to_limb.txt'
        args[f'model.arrowlimbs_to_limb-{sd}'] = f'{sd}-arrowlimbs_to_limb.txt'
        args[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-arrows_to_matchnext.txt'
        args[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-arrows_to_matchprev.txt'


def predict_limbs_ml(
    cs: ChartStruct,
    suite_singles: ModelSuite,
    suite_doubles: ModelSuite,
) -> None:
    """Predict limbs with ML ModelSuite + naturalness post-process.

    Replaces predict_limbs_pattern_only. Writes into cs.df['Limb annotation'].
    """
    pred_coords = cs.get_prediction_coordinates()
    if not pred_coords:
        return

    suite = suite_singles if cs.singles_or_doubles() == 'singles' else suite_doubles

    try:
        _, _, pred_limbs = ml_predict(cs, suite)
    except Exception as e:
        logger.warning(f'ML predict failed: {e}; falling back to rule-based')
        predict_limbs_pattern_only(cs)
        return

    pred_limbs = _fix_multihits_by_naturalness(pred_limbs, pred_coords)

    int_to_limb = {0: 'l', 1: 'r'}
    limb_strs = [int_to_limb[int(x)] for x in pred_limbs]
    cs.add_limb_annotations(pred_coords, limb_strs, 'Limb annotation')


def fuzzy_match_song_name(query: str, targets: list[str]) -> str | None:
    close_matches = difflib.get_close_matches(query, targets, n=1, cutoff=0.7)
    if len(close_matches) > 0:
        return close_matches[0]
    return None

VIS_SS_DIR = '/home/rodrigo/dev/piu/piu-vis-ss_for_piumx/public/chart-jsons/120524'


def load_vis_shortnames(vis_dir: str) -> set[str]:
    """Return set of shortnames that have a vis-ss JSON (ground truth exists)."""
    shortnames = set()
    for fn in os.listdir(vis_dir):
        if fn.endswith('.json'):
            shortnames.add(fn[:-5])  # strip .json
    return shortnames


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_missing_visss', action='store_true',
                        help='Only process charts that lack a vis-ss ground truth file')
    pargs = parser.parse_args()

    db_json_path = '/home/rodrigo/dev/piu/ligas-piu-api/master_db.json'
    simfiles_folder = '/home/rodrigo/dev/piu/piu_sim_files/'
    output_dir = '/home/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/processed_db'

    os.makedirs(output_dir, exist_ok=True)

    vis_shortnames = set()
    if pargs.only_missing_visss:
        vis_shortnames = load_vis_shortnames(VIS_SS_DIR)
        logger.info(f'Loaded {len(vis_shortnames)} vis-ss shortnames — will skip these charts')

    # 0. Load ML models once
    setup_model_args(MODEL_DIR)
    logger.info(f'Loading ML ModelSuites from {MODEL_DIR} ...')
    suite_singles = ModelSuite('singles')
    suite_doubles = ModelSuite('doubles')
    logger.info('ML models loaded.')

    # 1. Load DB data
    logger.info("Loading master DB...")
    with open(db_json_path, 'r', encoding='utf-8') as f:
        master_db = json.load(f)

    # 2. Crawl .ssc files
    logger.info("Crawling stepcharts from .ssc files...")
    stepcharts = crawl_stepcharts(simfiles_folder, skip_packs=[])
    
    # group by lowercased title
    song_name_to_stepcharts = defaultdict(list)
    for sc in stepcharts:
        title = sc['TITLE'].lower()
        song_name_to_stepcharts[title].append(sc)

    target_songs = list(song_name_to_stepcharts.keys())
    logger.info(f"Found {len(target_songs)} unique song titles in .ssc files.")

    match_count = 0
    processed_count = 0

    # 3. Process each song in DB
    for song in tqdm(master_db, desc="Processing DB songs"):
        query = song['song_name'].lower()
        matched_title = fuzzy_match_song_name(query, target_songs)
        
        if not matched_title:
            logger.debug(f"Could not match DB song: {song['song_name']}")
            continue
        
        match_count += 1
        ssc_stepcharts = song_name_to_stepcharts[matched_title]
        
        # Build index of level + mode for SSC stepcharts
        # SSC modes typically are 'singles', 'doubles', 'half-doubles', 'co-op', 'routine'
        ssc_idx = {}
        for sc in ssc_stepcharts:
            lvl = int(sc['METER']) if sc['METER'].isdigit() else 0
            style = sc['STEPSTYPE'].split('-')[-1] # e.g. pump-single -> single
            if style == 'single':
                mode = 'S'
            elif style == 'double':
                mode = 'D'
            elif style == 'halfdouble':
                mode = 'HD'
            elif style == 'routine':
                mode = 'routine'
            elif style == 'coop':
                mode = 'C'
            else:
                mode = style
            
            ssc_idx[(mode, lvl)] = sc

        for chart in song['charts']:
            mode = chart['mode'] # 'S', 'D', 'C'
            level = chart['level']
            
            sc = ssc_idx.get((mode, level))
            if not sc:
                continue
                
            # Process stepchart
            try:
                cs = ChartStruct.from_stepchart_ssc(sc)
                if cs is None:
                    continue

                if pargs.only_missing_visss and vis_shortnames:
                    shortname = cs.metadata.get('shortname', '')
                    if shortname in vis_shortnames:
                        continue  # has vis-ss ground truth — skip

                cs.annotate_time_since_downpress()
                cs.annotate_time_to_next_downpress()
                cs.annotate_line_repeats_previous()
                cs.annotate_line_repeats_next()
                cs.annotate_num_downpresses()
                cs.annotate_single_hold_ends_immediately()
                cs.df['Limb annotation'] = cs.init_limb_annotations()
                annotate_skills(cs)
                sections = segmentation(cs)

                predict_limbs_ml(cs, suite_singles, suite_doubles)
                cjs = ChartJsStruct.from_chartstruct(cs)

                # ── Build segment metadata ──────────────────────────────────────
                # Identify skill columns (boolean columns for each skill type)
                skill_cols = [c for c in cs.df.columns if c.startswith('__')
                              and not c.startswith('__time')
                              and not c.startswith('__line')
                              and not c.startswith('__num')
                              and not c.startswith('__single')]
                skill_names = [c.lstrip('_') for c in skill_cols]

                segment_list = []    # [[start_time, end_time, 0, 0], ...]
                segment_meta = []    # [{level, Skill badges, rare skills, ...}, ...]

                for sec in sections:
                    sec_df = cs.df.iloc[sec.start:sec.end]

                    # NPS = downpresses / duration
                    duration = max(sec.end_time - sec.start_time, 0.01)
                    num_dps = int(sec_df['__num downpresses'].sum()) if '__num downpresses' in sec_df.columns else 0
                    nps = round(num_dps / duration, 3)

                    # Skills present in this section
                    active_skills = []
                    for col, name in zip(skill_cols, skill_names):
                        if col in sec_df.columns and sec_df[col].any():
                            active_skills.append(name)

                    segment_list.append([
                        round(float(sec.start_time), 4),
                        round(float(sec.end_time), 4),
                        0, 0
                    ])
                    segment_meta.append({
                        'level': nps,           # relative difficulty via NPS
                        'Skill badges': active_skills,
                        'rare skills': [],
                        'Closest sections': [],
                    })

                # Compute chart-wide summary before normalising
                all_raw_nps = [m['level'] for m in segment_meta]
                total_duration = sum(
                    segment_list[i][1] - segment_list[i][0]
                    for i in range(len(segment_list))
                )
                nps_summary = round(sum(all_raw_nps) / max(len(all_raw_nps), 1), 2)
                peak_nps = round(max(all_raw_nps), 2) if all_raw_nps else 0.0

                # Chart-wide skill summary: union of all segment skill badges
                chart_skill_summary = sorted(set(
                    sk for m in segment_meta for sk in m['Skill badges']
                ))

                # Store raw NPS per segment for display before normalising
                for i, m in enumerate(segment_meta):
                    m['raw_nps'] = round(m['level'], 3)

                # Normalise level to 0–1 within the chart for colour coding
                if segment_meta:
                    min_nps, max_nps = min(all_raw_nps), max(all_raw_nps)
                    nps_range = max(max_nps - min_nps, 0.01)
                    for m in segment_meta:
                        m['level'] = round((m['raw_nps'] - min_nps) / nps_range, 4)

                # Inject into cjs metadata so the visualizer can read it
                cjs.metadata['Segments'] = segment_list
                cjs.metadata['Segment metadata'] = segment_meta
                cjs.metadata['Hold ticks'] = cjs.metadata.get('Hold ticks', [])
                cjs.metadata['eNPS annotations'] = []
                cjs.metadata['eNPS timeline data'] = []
                cjs.metadata['eNPS ranges of interest'] = []
                cjs.metadata['METER'] = level
                cjs.metadata['chart_skill_summary'] = chart_skill_summary
                cjs.metadata['nps_summary'] = nps_summary
                cjs.metadata['peak_nps'] = peak_nps

                out_path = os.path.join(output_dir, f"{song['id']}_{chart['id']}.json")

                wrapper = {
                    "song_id": song['id'],
                    "chart_id": chart['id'],
                    "song_name": song['song_name'],
                    "mode": mode,
                    "level": level,
                    "cjs": cjs.get_json_struct(),
                    "sections": [
                        {
                            "start_time": round(float(s.start_time), 4),
                            "end_time": round(float(s.end_time), 4),
                        } for s in sections
                    ]
                }

                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(wrapper, f)

                processed_count += 1
            except Exception as e:
                logger.warning(f"Skipping {song['song_name']} {mode}{level}: {e}")

    logger.success(f"Matched {match_count}/{len(master_db)} songs.")
    logger.success(f"Processed and exported {processed_count} charts.")


if __name__ == '__main__':
    main()
