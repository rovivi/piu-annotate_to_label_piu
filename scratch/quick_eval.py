from __future__ import annotations
import os
import pandas as pd
from loguru import logger
from tqdm import tqdm
from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.models import ModelSuite
from piu_annotate.ml.predictor import predict
from piu_annotate.ml import featurizers
from hackerargs import args

MODEL_DIR = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss'

def setup_model_args(model_dir: str) -> None:
    args['model'] = 'lightgbm'
    args['model.dir'] = model_dir
    for sd in ('singles', 'doubles'):
        args[f'model.arrows_to_limb-{sd}'] = f'{sd}-arrows_to_limb.txt'
        args[f'model.arrowlimbs_to_limb-{sd}'] = f'{sd}-arrowlimbs_to_limb.txt'
        args[f'model.arrows_to_matchnext-{sd}'] = f'{sd}-arrows_to_matchnext.txt'
        args[f'model.arrows_to_matchprev-{sd}'] = f'{sd}-arrows_to_matchprev.txt'

def accuracy(fcs, pred_limbs):
    eval_d = fcs.evaluate(pred_limbs, verbose = False)
    return eval_d['accuracy-float']

def run_sample(file_list, sd):
    setup_model_args(MODEL_DIR)
    model_suite = ModelSuite(sd)
    with open(file_list) as f:
        csvs = [line.strip() for line in f if line.strip()]
    
    accuracies = []
    for csv in tqdm(csvs):
        try:
            cs = ChartStruct.from_file(csv)
            cs, fcs, pred_limbs = predict(cs, model_suite)
            accuracies.append(accuracy(fcs, pred_limbs))
        except:
            continue
    
    if accuracies:
        avg = sum(accuracies) / len(accuracies)
        print(f"Average Accuracy for {sd}: {avg:.2%}")
    else:
        print(f"No results for {sd}")

if __name__ == "__main__":
    import sys
    run_sample(sys.argv[1], sys.argv[2])
