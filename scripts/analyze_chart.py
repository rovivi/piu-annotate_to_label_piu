#!/usr/bin/env python3
"""Quick script to analyze a chart from command line."""

import argparse
import sys
import json

from piu_annotate.formats.sscfile import SongSSC
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.segment.skills import annotate_skills
from piu_annotate.segment.segment import Segmenter
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict as predict_limbs
from piu_annotate.formats.jsplot import ChartJsStruct


def main():
    parser = argparse.ArgumentParser(description='Analyze a PIU chart')
    parser.add_argument('ssc_file', help='Path to .ssc file')
    parser.add_argument('--difficulty', type=int, default=None, help='Chart difficulty level')
    parser.add_argument('--play-style', default='singles', choices=['singles', 'doubles'])
    parser.add_argument('--models', default='./models', help='Path to models directory')
    parser.add_argument('--output', default=None, help='Output JSON path')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        print(f"Loading {args.ssc_file}...")

    song = SongSSC.from_file(args.ssc_file)

    if args.difficulty is not None:
        stepchart = song.get_stepchart(difficulty=args.difficulty, play_style=args.play_style)
    else:
        stepchart = song.stepcharts[0]

    cs = ChartStruct.from_stepchart_ssc(stepchart)

    if args.verbose:
        print(f"Chart loaded: {cs.singles_or_doubles()} S{cs.get_chart_level()}")
        print(f"Lines: {len(cs.df)}")

    skills = ['drill', 'run', 'bracket', 'twist_90', 'jack', 'footswitch']
    annotate_skills(cs, skill_names=skills)

    segmenter = Segmenter()
    sections = segmenter.segmentation(cs)

    if args.verbose:
        print(f"Sections: {len(sections)}")

    try:
        model_suite = ModelSuite.load(args.models)
        cs, fcs, pred_limbs = predict_limbs(cs, model_suite, verbose=args.verbose)
        cs.add_limb_annotations(fcs.pred_coords, pred_limbs, 'Predicted limbs')

        if args.verbose:
            print(f"Predicted limbs: {len(pred_limbs)}")
    except Exception as e:
        if args.verbose:
            print(f"Could not load models: {e}")
        print("Warning: ML models not available, skipping limb prediction")

    result = {
        'source_file': args.ssc_file,
        'level': cs.get_chart_level(),
        'sord': cs.singles_or_doubles(),
        'num_lines': len(cs.df),
        'num_sections': len(sections),
        'skills': skills,
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        if args.verbose:
            print(f"Output written to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()