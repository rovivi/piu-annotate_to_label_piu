from __future__ import annotations
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray
from collections import defaultdict
import pandas as pd

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict

def setup_model_args(model_dir: str, model_type: str = 'lightgbm') -> None:
    args['model'] = model_type
    args['model.dir'] = model_dir
    for sd in ('singles', 'doubles'):
        if model_type == 'lightgbm':
            args[f'model.arrows_to_limb-{sd}'] = f'{sd}-arrows_to_limb.txt'
            args[f'model.arrowlimbs_to_limb-{sd}'] = f'{sd}-arrowlimbs_to_limb.txt'
            args[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-arrows_to_matchnext.txt'
            args[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-arrows_to_matchprev.txt'
        else:
            args[f'model.arrows_to_limb-{sd}'] = f'{sd}-mlx_model.safetensors'
            args[f'model.arrowlimbs_to_limb-{sd}'] = f'{sd}-mlx_model.safetensors'
            args[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-mlx_model.safetensors'
            args[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-mlx_model.safetensors'

def accuracy(fcs: featurizers.ChartStructFeaturizer, pred_limbs: NDArray):
    eval_d = fcs.evaluate(pred_limbs, verbose = False)
    return eval_d['accuracy-float']

def main():
    model_type = args.get('model', 'lightgbm')
    singles_or_doubles = args.get('singles_or_doubles', 'singles')
    
    if args.get('model_dir'):
        model_dir = args['model_dir']
    else:
        model_dir = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/' + ('visss' if model_type == 'lightgbm' else 'mlx')
    
    setup_model_args(model_dir, model_type)
    
    if args.get('chart_struct_csv'):
        csv = args['chart_struct_csv']
        logger.info(f'Using {csv=}')
        cs: ChartStruct = ChartStruct.from_file(csv)
        model_suite = ModelSuite(cs.singles_or_doubles())
        cs, fcs, pred_limbs = predict(cs, model_suite)
        acc = accuracy(fcs, pred_limbs)
        logger.success(f'Accuracy: {acc:.2%}')
        
        # Save results
        basename = os.path.basename(csv)
        os.makedirs('temp', exist_ok=True)
        out_fn = f'temp/{basename}'
        cs.to_csv(out_fn)
        logger.info(f'Saved to {out_fn}')
    else:
        csv_folder = args.get('manual_chart_struct_folder', 'artifacts/manual-chartstructs/visss-120524/')
        logger.info(f'Running {singles_or_doubles} ...')
        model_suite = ModelSuite(singles_or_doubles)

        csvs = []
        for dirpath, _, files in os.walk(csv_folder):
            for file in files:
                if file.endswith('.csv') and 'exclude' not in dirpath:
                    csvs.append(os.path.join(dirpath, file))
        
        logger.info(f'Found {len(csvs)} csvs ...')
        
        dd = defaultdict(list)
        for csv in tqdm(csvs):
            try:
                cs = ChartStruct.from_file(csv)
                if cs.singles_or_doubles() != singles_or_doubles:
                    continue
                cs, fcs, pred_limbs = predict(cs, model_suite)
                dd['File'].append(os.path.basename(csv))
                dd['Accuracy'].append(accuracy(fcs, pred_limbs))
            except Exception as e:
                logger.warning(f'Error processing {csv}: {e}')

        if dd['Accuracy']:
            stats_df = pd.DataFrame(dd)
            os.makedirs('temp', exist_ok=True)
            stats_df.to_csv(f'temp/stats-{singles_or_doubles}.csv')
            logger.info(stats_df['Accuracy'].describe())
        else:
            logger.warning('No compatible charts found.')

    logger.success('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chart_struct_csv', default=None)
    parser.add_argument('--manual_chart_struct_folder', default='artifacts/manual-chartstructs/visss-120524/')
    parser.add_argument('--singles_or_doubles', default='singles')
    parser.add_argument('--model', default='lightgbm')
    parser.add_argument('--model_dir', default=None)
    args.parse_args(parser)
    main()