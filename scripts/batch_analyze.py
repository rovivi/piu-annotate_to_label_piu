from __future__ import annotations
#!/usr/bin/env python3
"""Batch processing script for multiple charts."""

import argparse
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from piu_annotate.formats.sscfile import SongSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.segment.segment import Segmenter
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict as predict_limbs
from piu_annotate.agent_tools import export_chart_summary


def process_single_chart(args_tuple):
    ssc_path, models_dir, skills, skip_models = args_tuple
    try:
        song = SongSSC.from_file(ssc_path)
        stepchart = song.stepcharts[0]
        cs = ChartStruct.from_stepchart_ssc(stepchart)

        annotate_skills(cs, skill_names=skills)

        segmenter = Segmenter()
        sections = segmenter.segmentation(cs)

        pred_limbs = None
        if not skip_models:
            try:
                model_suite = ModelSuite.load(models_dir)
                cs, fcs, pred_limbs = predict_limbs(cs, model_suite)
                cs.add_limb_annotations(fcs.pred_coords, pred_limbs, 'Predicted limbs')
            except Exception:
                pass

        summary = export_chart_summary(cs)
        summary['sections'] = len(sections)
        summary['success'] = True
        return summary
    except Exception as e:
        return {'source_file': ssc_path, 'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Batch analyze PIU charts')
    parser.add_argument('input_dir', help='Directory containing .ssc files')
    parser.add_argument('--models', default='./models', help='Path to models directory')
    parser.add_argument('--output', default='batch_results.json', help='Output JSON file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--skip-models', action='store_true', help='Skip ML limb prediction')
    parser.add_argument('--skills', nargs='+',
                        default=['drill', 'run', 'bracket', 'twist_90', 'jack', 'footswitch'],
                        help='Skills to annotate')

    args = parser.parse_args()

    ssc_files = glob(os.path.join(args.input_dir, '**', '*.ssc'), recursive=True)
    print(f"Found {len(ssc_files)} .ssc files")

    tasks = [(f, args.models, args.skills, args.skip_models) for f in ssc_files]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_chart, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"[{i+1}/{len(ssc_files)}] {result.get('source_file', 'unknown')}: "
                  f"success={result.get('success', False)}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")

    success_count = sum(1 for r in results if r.get('success', False))
    print(f"Success: {success_count}/{len(results)}")


if __name__ == '__main__':
    main()