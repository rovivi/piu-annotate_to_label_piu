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

MODEL_DIR = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss'

def setup_model_args(model_dir: str) -> None:
    args['model'] = 'lightgbm'
    args['model.dir'] = model_dir
    for sd in ('singles', 'doubles'):
        args[f'model.arrows_to_limb-{sd}'] = f'{sd}-arrows_to_limb.txt'
        args[f'model.arrowlimbs_to_limb-{sd}'] = f'{sd}-arrowlimbs_to_limb.txt'
        args[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-arrows_to_matchnext.txt'
        args[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-arrows_to_matchprev.txt'

def accuracy(fcs: featurizers.ChartStructFeaturizer, pred_limbs: NDArray):
    eval_d = fcs.evaluate(pred_limbs, verbose = False)
    return eval_d['accuracy-float']

def main():
    setup_model_args(MODEL_DIR)
    
    if not args.get('run_folder'):
        csv = args.get('chart_struct_csv')
        if not csv:
            logger.error('Must provide --chart_struct_csv if not using --run_folder')
            return
        logger.info(f'Using {csv=}')
        cs: ChartStruct = ChartStruct.from_file(csv)
        singles_or_doubles = cs.singles_or_doubles()
        model_suite = ModelSuite(singles_or_doubles)

        cs, fcs, pred_limbs = predict(cs, model_suite, verbose = True)

        # annotate
        pred_coords = cs.get_prediction_coordinates()
        int_to_limb = {0: 'l', 1: 'r'}
        pred_limb_strs = [int_to_limb[i] for i in pred_limbs]
        cs.add_limb_annotations(pred_coords, pred_limb_strs, '__pred limb final')

        cs.df['Error'] = (
            cs.df['__pred limb final'] != cs.df['Limb annotation']
        ).astype(int)

        basename = os.path.basename(csv)
        os.makedirs('temp', exist_ok=True)
        out_fn = f'temp/{basename}'
        cs.to_csv(out_fn)
        logger.info(f'Saved to {out_fn}')
    else:
        csv_folder = args['manual_chart_struct_folder']
        singles_or_doubles = args['singles_or_doubles']
        logger.info(f'Running {singles_or_doubles} ...')
        model_suite = ModelSuite(singles_or_doubles)

        # crawl all subdirs for csvs
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
    parser.add_argument('--run_folder', type=bool, default=False)
    args.parse_args(parser)
    main()