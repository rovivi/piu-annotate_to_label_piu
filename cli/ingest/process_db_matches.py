import json
import os
import difflib
import traceback
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import numpy as np

from piu_annotate.crawl import crawl_stepcharts
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.formats.jsplot import ChartJsStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.segment.segment import segmentation
from piu_annotate.reasoning.reasoners import PatternReasoner


def predict_limbs_pattern_only(cs: ChartStruct) -> None:
    """ Predict limbs using PatternReasoner (rule-based, no ML models).
        Fills in unresolved positions with simple left-right alternation.
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

    int_to_limb = {0: 'l', 1: 'r'}
    limb_strs = [int_to_limb[int(x)] for x in pred_limbs]
    cs.add_limb_annotations(pred_coords, limb_strs, 'Limb annotation')

def fuzzy_match_song_name(query: str, targets: list[str]) -> str | None:
    close_matches = difflib.get_close_matches(query, targets, n=1, cutoff=0.7)
    if len(close_matches) > 0:
        return close_matches[0]
    return None

def main():
    db_json_path = '/home/rodrigo/dev/piu/ligas-piu-api/master_db.json'
    simfiles_folder = '/home/rodrigo/dev/piu/piu_sim_files/'
    output_dir = '/home/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/processed_db'

    os.makedirs(output_dir, exist_ok=True)

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

                cs.annotate_time_since_downpress()
                cs.annotate_time_to_next_downpress()
                cs.annotate_line_repeats_previous()
                cs.annotate_line_repeats_next()
                cs.annotate_num_downpresses()
                cs.annotate_single_hold_ends_immediately()
                cs.df['Limb annotation'] = cs.init_limb_annotations()
                annotate_skills(cs)
                sections = segmentation(cs)

                predict_limbs_pattern_only(cs)
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
