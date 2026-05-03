from __future__ import annotations
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from hackerargs import args

from piu_annotate.formats.chart import ChartStruct
from piu_annotate.ml.featurizers import ChartStructFeaturizer
from piu_annotate.ml.mlx_architecture import LimbTransformer

def load_dataset(folder: str, sd: str, with_limbs: bool, limit: int = None):
    files = list(Path(folder).glob('*.csv'))
    
    all_x = []
    all_y = []
    
    logger.info(f"Loading files for {sd} (with_limbs={with_limbs})...")
    count = 0
    for f in tqdm(files):
        try:
            cs = ChartStruct.from_file(str(f))
            if cs.singles_or_doubles() != sd:
                continue
            
            ft = ChartStructFeaturizer(cs)
            y = ft.get_labels_from_limb_col('Limb annotation') # N x 1
            if with_limbs:
                x = ft.featurize_arrowlimbs_with_context(y.squeeze()) # N x D_augmented
            else:
                x = ft.featurize_arrows_with_context() # N x D
            
            all_x.append(x)
            all_y.append(y)
            count += 1
            if limit and count >= limit:
                break
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")
            continue
            
    return np.concatenate(all_x), np.concatenate(all_y)

def train_model(X, Y, model_path):
    input_dim = X.shape[1]
    model = LimbTransformer(input_dim=input_dim)
    mx.eval(model.parameters())
    
    optimizer = optim.Adam(learning_rate=1e-3)
    
    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.binary_cross_entropy(logits, y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    batch_size = 64
    num_epochs = 3 # Fast for demo
    
    for epoch in range(num_epochs):
        perm = np.random.permutation(len(X))
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            idxs = perm[i : i + batch_size]
            x_batch = mx.array(X[idxs])[:, None, :]
            y_batch = mx.array(Y[idxs])[:, None, None]
            
            loss, grads = loss_and_grad(model, x_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}, Loss: {epoch_loss / (len(X)/batch_size):.4f}")

    model.save_weights(model_path)
    logger.success(f"Model saved to {model_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--singles_or_doubles', type=str)
    parser.add_argument('--manual_chart_struct_folder', type=str)
    parser.add_argument('--out_dir', type=str)
    args.parse_args(parser)
    
    sd = args['singles_or_doubles']
    folder = args['manual_chart_struct_folder']
    out_dir = args['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    # 1. Train Arrows-to-Limb
    X, Y = load_dataset(folder, sd, with_limbs=False, limit=4000)
    train_model(X, Y, os.path.join(out_dir, f"{sd}-arrows_to_limb-mlx_model.safetensors"))
    
    # 2. Train ArrowLimbs-to-Limb
    X, Y = load_dataset(folder, sd, with_limbs=True, limit=4000)
    train_model(X, Y, os.path.join(out_dir, f"{sd}-arrowlimbs_to_limb-mlx_model.safetensors"))

if __name__ == "__main__":
    main()
