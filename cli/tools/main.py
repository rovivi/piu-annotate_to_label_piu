#!/usr/bin/env python3
"""
Unified CLI for piu-annotate

Usage:
    python -m cli.tools.analyze <ssc_file> [options]
    python -m cli.tools.batch <directory> [options]
    python -m cli.tools.info <csv_file>
"""

import argparse
import json
import sys

from piu_annotate.formats.sscfile import SongSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.segment.segment import Segmenter
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict as predict_limbs
from piu_annotate.formats.jsplot import ChartJsStruct


def cmd_analyze(args):
    song = SongSSC.from_file(args.ssc_file)

    if args.difficulty is not None:
        stepchart = song.get_stepchart(difficulty=args.difficulty, play_style=args.play_style)
    else:
        stepchart = song.stepcharts[0]

    cs = ChartStruct.from_stepchart_ssc(stepchart)

    skills = args.skills.split(',') if args.skills else ['drill', 'run', 'bracket', 'twist_90', 'jack']
    annotate_skills(cs, skill_names=skills)

    if args.segment:
        segmenter = Segmenter()
        sections = segmenter.segmentation(cs)

    if args.predict:
        try:
            model_suite = ModelSuite.load(args.models)
            cs, fcs, pred_limbs = predict_limbs(cs, model_suite, verbose=args.verbose)
            cs.add_limb_annotations(fcs.pred_coords, pred_limbs, 'Predicted limbs')
        except Exception as e:
            print(f"Warning: Could not load models: {e}", file=sys.stderr)

    if args.output_json:
        cjs = ChartJsStruct.from_chartstruct(cs)
        cjs.to_json(args.output_json)
        print(f"JSON output written to {args.output_json}")

    result = {
        'source': args.ssc_file,
        'level': cs.get_chart_level(),
        'sord': cs.singles_or_doubles(),
        'lines': len(cs.df),
        'duration': round(cs.df['Time'].max() - cs.df['Time'].min(), 2),
    }
    if args.segment:
        result['sections'] = len(sections)
    print(json.dumps(result, indent=2))


def cmd_batch(args):
    from glob import glob
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ssc_files = glob(f"{args.input_dir}/**/*.ssc", recursive=True)
    print(f"Found {len(ssc_files)} .ssc files")

    def process_one(ssc_path):
        try:
            song = SongSSC.from_file(ssc_path)
            stepchart = song.stepcharts[0]
            cs = ChartStruct.from_stepchart_ssc(stepchart)
            return {'file': ssc_path, 'level': cs.get_chart_level(), 'sord': cs.singles_or_doubles(), 'success': True}
        except Exception as e:
            return {'file': ssc_path, 'error': str(e), 'success': False}

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, f) for f in ssc_files]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result.get('success') else "FAIL"
            print(f"[{i+1}/{len(ssc_files)}] {result.get('file')} [{status}]")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


def cmd_info(args):
    cs = ChartStruct.from_file(args.csv_file)
    print(f"Source: {cs.source_file}")
    print(f"Level: {cs.get_chart_level()}")
    print(f"Type: {cs.singles_or_doubles()}")
    print(f"Lines: {len(cs.df)}")
    print(f"Duration: {cs.df['Time'].max() - cs.df['Time'].min():.2f}s")

    skill_cols = [c for c in cs.df.columns if cs.df[c].dtype == bool and c not in ['Metadata']]
    if skill_cols:
        print(f"Skills annotated: {len(skill_cols)}")
        for col in skill_cols[:10]:
            count = cs.df[col].sum()
            print(f"  {col}: {count}")

    if 'Limb annotation' in cs.df.columns and cs.df['Limb annotation'].iloc[0]:
        print("Limb annotations: Present")


def main():
    parser = argparse.ArgumentParser(description='piu-annotate CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    p_analyze = subparsers.add_parser('analyze', help='Analyze a single chart')
    p_analyze.add_argument('ssc_file', help='Path to .ssc file')
    p_analyze.add_argument('-d', '--difficulty', type=int, help='Difficulty level')
    p_analyze.add_argument('-p', '--play-style', default='singles', choices=['singles', 'doubles'])
    p_analyze.add_argument('--models', default='./models', help='Models directory')
    p_analyze.add_argument('--skills', help='Comma-separated skill list')
    p_analyze.add_argument('--segment', action='store_true', help='Segment chart')
    p_analyze.add_argument('--predict', action='store_true', help='Predict limbs')
    p_analyze.add_argument('--output-json', help='Output JSON path')
    p_analyze.add_argument('-v', '--verbose', action='store_true')

    p_batch = subparsers.add_parser('batch', help='Batch process directory')
    p_batch.add_argument('input_dir', help='Input directory')
    p_batch.add_argument('-o', '--output', default='batch_results.json')
    p_batch.add_argument('-w', '--workers', type=int, default=4)

    p_info = subparsers.add_parser('info', help='Show chart info')
    p_info.add_argument('csv_file', help='Path to ChartStruct CSV')

    args = parser.parse_args()
    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'batch':
        cmd_batch(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()