from __future__ import annotations
import os
import shutil
import pandas as pd
from loguru import logger
from tqdm import tqdm
from hackerargs import args
import argparse

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.tactics import Tactician
from piu_annotate.ml.featurizers import ChartStructFeaturizer
from piu_annotate.reasoning.reasoners import PatternReasoner

def setup_model_args(model_dir: str, model_type: str = 'mlx') -> None:
    args['model'] = model_type
    args['model.dir'] = model_dir
    for sd in ('singles', 'doubles'):
        args[f'model.arrows_to_limb-{sd}'] = f'{sd}-arrows_to_limb-mlx_model.safetensors'
        args[f'model.arrowlimbs_to_limb-{sd}'] = f'{sd}-arrowlimbs_to_limb-mlx_model.safetensors'
        args[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-arrows_to_limb-mlx_model.safetensors'
        args[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-arrows_to_limb-mlx_model.safetensors'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='artifacts/manual-chartstructs/visss-120524/')
    parser.add_argument('--output_folder', default='artifacts/trouble-charts/')
    parser.add_argument('--min_level', type=int, default=20)
    parser.add_argument('--top_n', type=int, default=30)
    parser.add_argument('--model_dir', default='artifacts/models/mlx/')
    pargs = args.parse_args(parser)

    setup_model_args(pargs.model_dir, 'mlx')
    os.makedirs(pargs.output_folder, exist_ok=True)
    
    csvs = []
    for dirpath, _, files in os.walk(pargs.input_folder):
        for file in files:
            if file.endswith('.csv'):
                csvs.append(os.path.join(dirpath, file))
    
    logger.info(f"Filtering {len(csvs)} charts for Level >= {pargs.min_level}...")
    
    heavy_charts = []
    for csv in tqdm(csvs, desc="Filtering Levels"):
        try:
            cs = ChartStruct.from_file(csv)
            if cs.get_chart_level() >= pargs.min_level:
                heavy_charts.append((csv, cs))
        except:
            continue
            
    logger.info(f"Found {len(heavy_charts)} heavy charts. Analyzing discrepancies (Fast Mode)...")
    
    results = []
    suites = {'singles': ModelSuite('singles'), 'doubles': ModelSuite('doubles')}

    for csv, cs in tqdm(heavy_charts, desc="Predicting (Fast)"):
        try:
            sd = cs.singles_or_doubles()
            model_suite = suites[sd]
            
            fcs = ChartStructFeaturizer(cs)
            reasoner = PatternReasoner(cs, verbose=False)
            tactics = Tactician(cs, fcs, model_suite, verbose=False)
            
            # Fast prediction without beam search
            pred_limbs, abstained = reasoner.propose_limbs()
            pred_limbs = tactics.initial_predict(pred_limbs, abstained)
            
            eval_d = fcs.evaluate(pred_limbs, verbose=False)
            acc = eval_d['accuracy-float']
            results.append({
                'csv': csv,
                'level': cs.get_chart_level(),
                'accuracy': acc,
                'sd': sd
            })
        except:
            continue
            
    results.sort(key=lambda x: x['accuracy'])
    
    for i, res in enumerate(results[:pargs.top_n]):
        basename = os.path.basename(res['csv'])
        dest = os.path.join(pargs.output_folder, basename)
        shutil.copy(res['csv'], dest)
        logger.info(f"[{i+1}] {basename} - Acc: {res['accuracy']:.2%} (L{res['level']})")

    logger.success(f"Trouble dataset created in {pargs.output_folder}")

if __name__ == "__main__":
    main()
