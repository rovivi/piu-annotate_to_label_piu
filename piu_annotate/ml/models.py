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
import pickle
from numpy.typing import NDArray
from operator import itemgetter
import itertools

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import Booster
import mlx.core as mx
from piu_annotate.ml.mlx_architecture import LimbTransformer


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


class ModelSuite:
    def __init__(self, singles_or_doubles: str):
        """ Stores suite of ML models for limb prediction """
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
            is_arrow_only = 'arrows_to_limb' in model_name or 'match' in model_name
            suffix = 'arrows_to_limb' if is_arrow_only else 'arrowlimbs_to_limb'
            model_file = os.path.join(args['model.dir'], f"{self.singles_or_doubles}-{suffix}-mlx_model.safetensors")
            
            if self.singles_or_doubles == 'singles':
                input_dim = 698 if is_arrow_only else 755
            else:
                input_dim = 903 if is_arrow_only else 960
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
    def __init__(self, model: LimbTransformer):
        self.model = model

    @staticmethod
    def load(file: str, input_dim: int):
        model = LimbTransformer(input_dim=input_dim)
        model.load_weights(file)
        return MLXModel(model)

    def predict(self, points: NDArray) -> NDArray:
        x = mx.array(points).astype(mx.float32)
        if len(x.shape) == 2:
            x = x[None] # Add batch dim
        out = self.model(x)
        # out shape: (batch, seq, 1)
        prob = mx.sigmoid(out)
        return np.array((prob > 0.5).astype(mx.int32)).squeeze()

    def predict_prob(self, points: NDArray) -> NDArray:
        x = mx.array(points).astype(mx.float32)
        if len(x.shape) == 2:
            x = x[None]
        out = self.model(x)
        p = np.array(mx.sigmoid(out)).squeeze()
        if len(p.shape) == 0:
            p = p.item()
            return np.array([[1 - p, p]])
        return np.stack([1 - p, p]).T
    
    def predict_log_prob(self, points: NDArray) -> NDArray:
        return np.log(self.predict_prob(points))
    
