from __future__ import annotations
"""
    Model
"""
import argparse
import os
from hackerargs import args
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import functools
import pickle
from numpy.typing import NDArray
from operator import itemgetter
import itertools

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import Booster
import mlx.core as mx
from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer


supported_models = ['lightgbm', 'mlx']


class ModelWrapper:
    def __init__(self):
        pass

    def predict(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 1 binary array of 0 or 1 """
        raise NotImplementedError    

    def predict_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of p(0) and p(1) """
        raise NotImplementedError    
    
    def predict_log_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of logp(0) and logp(1) """
        raise NotImplementedError    


class DummyMatchModel(ModelWrapper):
    def predict(self, points: NDArray) -> NDArray:
        return np.zeros(len(points), dtype=int)
    def predict_prob(self, points: NDArray) -> NDArray:
        return np.ones((len(points), 2)) * 0.5
    def predict_log_prob(self, points: NDArray) -> NDArray:
        return np.zeros((len(points), 2))

class ModelSuite:
    def __init__(self, singles_or_doubles: str):
        self.singles_or_doubles = singles_or_doubles
        model_type = args['model']
        assert model_type in supported_models, f'{model_type=} not in {supported_models=}'
        self.model_type = model_type

        sd = self.singles_or_doubles
        self.model_arrows_to_limb = self.load(f'model.arrows_to_limb-{sd}')
        self.model_arrowlimbs_to_limb = self.load(f'model.arrowlimbs_to_limb-{sd}')
        self.model_arrows_to_matchnext = self.load(f'model.arrows_to_matchnext-{sd}')
        self.model_arrows_to_matchprev = self.load(f'model.arrows_to_matchprev-{sd}')

    def load(self, model_name: str) -> ModelWrapper:
        model_file = os.path.join(args['model.dir'], args[model_name])
        if self.model_type == 'lightgbm':
            model = LGBModel.load(model_file)
        elif self.model_type == 'mlx':
            if 'match' in model_name:
                return DummyMatchModel()
            
            is_arrow_only = 'arrows_to_limb' in model_name
            suffix = 'arrows_to_limb' if is_arrow_only else 'arrowlimbs_to_limb'
            model_file = os.path.join(args['model.dir'], f"{self.singles_or_doubles}-{suffix}-mlx-best.safetensors")
            
            if self.singles_or_doubles == 'singles':
                input_dim = 18
            else:
                input_dim = 23
            
            # Note: We didn't train arrowlimbs_to_limb for MLX yet, but fallback to arrows_to_limb for now
            if not is_arrow_only:
                model_file = os.path.join(args['model.dir'], f"{self.singles_or_doubles}-arrows_to_limb-mlx-best.safetensors")
                
            model = MLXModel.load(model_file, input_dim=input_dim)
        return model


class LGBModel(ModelWrapper):
    def __init__(self, bst: Booster):
        self.bst = bst

    @staticmethod
    def load(file: str):
        return LGBModel(lgb.Booster(model_file = file))

    @staticmethod
    def train(points: NDArray, labels: NDArray):
        train_x, test_x, train_y, test_y = train_test_split(points, labels)

        train_data = lgb.Dataset(train_x, label = train_y)
        test_data = lgb.Dataset(test_x, label = test_y)
        params = {'objective': 'binary', 'metric': 'binary_logloss'}
        bst = lgb.train(params, train_data, valid_sets = [test_data])
        return LGBModel(bst)

    def save(self, file: str) -> None:
        self.bst.save_model(file)

    def predict(self, points: NDArray) -> NDArray:
        """ For N points, returns N-length binary array of 0 or 1 """
        return self.bst.predict(points).round().astype(int)

    def predict_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of p(0) and p(1) """
        p = self.bst.predict(points)
        return np.stack([1 - p, p]).T
    
    def predict_log_prob(self, points: NDArray) -> NDArray:
        """ For N points, returns N x 2 array of logp(0) and logp(1) """
        return np.log(self.predict_prob(points))


class MLXModel(ModelWrapper):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def load(file: str, input_dim: int, n_classes: int = 3):
        from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer
        model = LimbSequenceTransformer(input_dim=input_dim)
        model.load_weights(file)
        return MLXModel(model)

    def predict(self, points: NDArray) -> NDArray:
        p = self.predict_prob(points)
        if p.shape[-1] > 2:
            # Force choice between Left (0) and Right (1)
            return np.argmax(p[:, :2], axis=-1)
        return np.argmax(p, axis=-1)

    def predict_prob(self, points: NDArray) -> NDArray:
        N = len(points)
        chunk_size = 512
        overlap = 128
        stride = chunk_size - overlap
        
        # Define compiled forward pass
        @functools.lru_cache(maxsize=None)
        def get_compiled_forward():
            def forward(x, mask):
                return mx.softmax(self.model(x, padding_mask=mask), axis=-1)
            return mx.compile(forward)
            
        compiled_forward = get_compiled_forward()

        if N <= chunk_size:
            # Pad to chunk_size
            x_padded = np.zeros((chunk_size, points.shape[1]), dtype=np.float32)
            x_padded[:N] = points
            x_mx = mx.array(x_padded)[None]
            
            mask_padded = np.ones((1, chunk_size), dtype=bool)
            mask_padded[0, :N] = False
            mask_mx = mx.array(mask_padded)
            
            # Forward pass: (1, L, 3)
            p = np.array(compiled_forward(x_mx, mask_mx))[0] # Remove B dim
            p = p[:N] # Keep only actual length
            return p

        p_total = np.zeros((N, 3), dtype=np.float32)
        weight_total = np.zeros((N, 1), dtype=np.float32)

        window = np.ones((chunk_size, 1), dtype=np.float32)
        window[:overlap, 0] = np.linspace(0, 1, overlap)
        window[-overlap:, 0] = np.linspace(1, 0, overlap)

        for start in range(0, N, stride):
            end = min(start + chunk_size, N)
            actual_len = end - start
            
            chunk_points = points[start:end]
            
            x_padded = np.zeros((chunk_size, points.shape[1]), dtype=np.float32)
            x_padded[:actual_len] = chunk_points
            x_mx = mx.array(x_padded)[None]
            
            mask_padded = np.ones((1, chunk_size), dtype=bool)
            mask_padded[0, :actual_len] = False
            mask_mx = mx.array(mask_padded)
            
            p = np.array(compiled_forward(x_mx, mask_mx))[0] # (L, 3)
            p = p[:actual_len]
            
            if actual_len == 1:
                p = p[None, :]

            w = window[:actual_len].copy()
            if start == 0:
                w[:overlap] = 1.0
            if end == N:
                w[-overlap:] = 1.0

            p_total[start:end] += p * w
            weight_total[start:end] += w
            
            if end == N:
                break
                
        p_total = p_total / np.maximum(weight_total, 1e-6)
        return p_total
    
    def predict_log_prob(self, points: NDArray) -> NDArray:
        return np.log(np.maximum(self.predict_prob(points), 1e-10))
    
