from __future__ import annotations
"""
    Featurize
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle
import hashlib
import gzip

import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml import featurizers
from piu_annotate.ml.datapoints import ArrowDataPoint


def md5_hash(tup) -> str:
    return hashlib.md5(pickle.dumps(tup)).hexdigest()


def guess_singles_or_doubles_from_filename(filename: str) -> str:
    """ """
    basename = os.path.basename(filename)
    sord = basename.split('_')[-2][0]
    if sord == 'S':
        return 'singles'
    elif sord == 'D':
        return 'doubles'
    return 'unsure'


def create_dataset(
    csvs: list[str],
    get_label_func: callable,
    dataset_name: str,
    use_limb_features: bool = False,
):
    singles_doubles = args.setdefault('singles_or_doubles', 'singles')

    # get hash
    hashid = md5_hash(( tuple(sorted(csvs)), dataset_name, use_limb_features, singles_doubles ))

    # try to load dataset
    dataset_storage = args.setdefault('dataset_storage', '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/cli/temp/dataset-storage/')
    storage_file = os.path.join(dataset_storage, f'{hashid}.pkl.gz')
    if os.path.isfile(storage_file) and not args.setdefault('rebuild_datasets', False):
        logger.info(f'Loading from {storage_file} ...')
        with gzip.open(storage_file, 'rb') as f:
            return pickle.load(f)

    # subset csvs to singles or doubles
    csv_sord = [csv for csv in csvs
                if guess_singles_or_doubles_from_filename(csv) in [singles_doubles, 'unsure']]

    # if singles_doubles == 'doubles':
    #     # limit n csvs to train on to prevent OOM killing
    #     max_n_csvs = 460
    #     if len(csv_sord) > max_n_csvs:
    #         np.random.shuffle(csv_sord)
    #         csv_sord = csv_sord[:max_n_csvs]
    #         logger.info(f'Randomly subsetted training to {len(csv_sord)} to prevent OOM kill')

    all_points, all_labels = [], []
    n_csvs = 0
    for csv in tqdm(csv_sord):
        try:
            cs = ChartStruct.from_file(csv)
            if cs.singles_or_doubles() != singles_doubles:
                continue

            fcs = featurizers.ChartStructFeaturizer(cs)
            labels = get_label_func(fcs)
        except Exception as e:
            logger.warning(f'Skipping {csv} due to error: {e}')
            continue

        if use_limb_features:
            points = fcs.featurize_arrowlimbs_with_context(labels)
            feature_names = fcs.get_arrowlimb_context_feature_names()
        else:
            points = fcs.featurize_arrows_with_context()
            feature_names = fcs.get_arrow_context_feature_names()

        all_points.append(points)
        all_labels.append(labels)
        n_csvs += 1
    logger.info(f'Featurized {n_csvs} ChartStruct csvs ...')

    points = np.concatenate(all_points)
    labels = np.concatenate(all_labels)
    logger.info(f'Found dataset shape {points.shape}')

    result = (points, labels, feature_names)
    # with gzip.open(storage_file, 'wb') as f:
    #     pickle.dump(result, f)
    # logger.info(f'Stored dataset into {storage_file}')
    return result


def train_categorical_model(
    points: NDArray,
    labels: NDArray,
    feature_names: list[str],
    model_name: str = '',
) -> tuple:
    """Returns (booster, train_acc, val_acc, evals_result)."""
    train_x, test_x, train_y, test_y = train_test_split(
        points, labels, test_size=0.1, random_state=0
    )

    cat_features = [fn for fn in feature_names if fn.startswith('cat.')]
    train_data = lgb.Dataset(
        train_x, label=train_y,
        feature_name=feature_names, categorical_feature=cat_features,
    )
    test_data = lgb.Dataset(
        test_x, label=test_y,
        feature_name=feature_names, categorical_feature=cat_features,
    )

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'force_col_wise': True,
        'num_leaves': 63,
        'learning_rate': 0.05,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }

    evals_result: dict = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.record_evaluation(evals_result),
        lgb.log_evaluation(period=100),
    ]

    bst = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )

    train_pred = bst.predict(train_x).round()
    train_acc = float(sum(train_pred == train_y) / len(train_y))
    val_pred = bst.predict(test_x).round()
    val_acc = float(sum(val_pred == test_y) / len(test_y))
    logger.info(f'{model_name}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  '
                f'best_iter={bst.best_iteration}')
    return bst, train_acc, val_acc, evals_result


def save_model(bst: Booster, name):
    out_dir = args.setdefault('out_dir', '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss')
    singles_doubles = args.setdefault('singles_or_doubles', 'singles')
    out_fn = os.path.join(out_dir, f'{singles_doubles}-{name}.txt')

    bst.save_model(out_fn)

    logger.info(f'Saved model to {out_fn}')
    return


def plot_training_curves(all_results: dict, singles_doubles: str, out_dir: str):
    """Save a learning-curve PNG for every trained sub-model."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning('matplotlib not available – skipping training curves plot')
        return

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(f'LightGBM training curves – {singles_doubles}', fontsize=12)

    for col, (model_name, info) in enumerate(all_results.items()):
        ax = axes[0][col]
        er = info['evals_result']
        train_loss = er.get('train', {}).get('binary_logloss', [])
        val_loss   = er.get('val',   {}).get('binary_logloss', [])
        iters = range(1, len(train_loss) + 1)
        ax.plot(iters, train_loss, label=f"train (final {train_loss[-1]:.4f})" if train_loss else 'train')
        ax.plot(iters, val_loss,   label=f"val   (final {val_loss[-1]:.4f})"   if val_loss   else 'val',
                linestyle='--')
        ax.set_title(f"{model_name}\ntrain {info['train_acc']:.3%}  val {info['val_acc']:.3%}")
        ax.set_xlabel('Boosting round')
        ax.set_ylabel('Binary logloss')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'training_curves_{singles_doubles}.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    logger.info(f'Training curves saved to {out_path}')


def main():
    folder = args['manual_chart_struct_folder']

    csvs = []
    dirpaths = set()
    for dirpath, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.csv') and 'exclude' not in dirpath:
                csvs.append(os.path.join(dirpath, file))
                dirpaths.add(dirpath)
    logger.info(f'Found {len(csvs)} csvs in {len(dirpaths)} directories ...')

    singles_doubles = args.setdefault('singles_or_doubles', 'singles')
    all_results: dict = {}

    tasks = [
        ('arrows_to_limb',    lambda fcs: fcs.get_labels_from_limb_col('Limb annotation'), False),
        ('arrowlimbs_to_limb',lambda fcs: fcs.get_labels_from_limb_col('Limb annotation'), True),
        ('arrows_to_matchnext', lambda fcs: fcs.get_label_matches_next('Limb annotation'), False),
        ('arrows_to_matchprev', lambda fcs: fcs.get_label_matches_prev('Limb annotation'), False),
    ]

    for name, label_func, use_limb in tasks:
        logger.info(f'--- {name} ---')
        points, labels, feature_names = create_dataset(csvs, label_func, name, use_limb_features=use_limb)
        bst, train_acc, val_acc, evals_result = train_categorical_model(
            points, labels, feature_names, model_name=name,
        )
        save_model(bst, name)
        all_results[name] = {
            'train_acc': train_acc, 'val_acc': val_acc, 'evals_result': evals_result,
        }

    # Print summary table
    logger.info('\n=== Training summary ===')
    print(f'\n  {"model":<30}  {"train_acc":>9}  {"val_acc":>8}')
    print(f'  {"-"*52}')
    for name, info in all_results.items():
        print(f'  {name:<30}  {info["train_acc"]:>9.4%}  {info["val_acc"]:>8.4%}')

    # Save curves
    out_dir = args.setdefault('out_dir', '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/models/visss')
    plot_training_curves(all_results, singles_doubles, out_dir)

    logger.success('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Train LightGBM model suite on chart struct with manual limb annotations.
    """)
    parser.add_argument(
        '--manual_chart_struct_folder',
        default = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/visss-120524/',
        # default = '/Users/rodrigo/dev/piu/piu-annotate_to_label_piu/artifacts/manual-chartstructs/piucenter-manual-090624/',
    )
    parser.add_argument(
        '--singles_or_doubles', 
        default = 'singles',
    )
    args.parse_args(parser)
    main()