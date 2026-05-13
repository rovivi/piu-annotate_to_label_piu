"""Backend-agnostic ModelSuite.

Three backends:
  - ``lightgbm``: GBDT, legacy. Uses sliding-window features.
  - ``mlx``:      Apple Silicon transformer. Uses sequence features.
  - ``torch``:    PyTorch transformer (CUDA / ROCm / MPS / CPU). Uses sequence features.

The suite is configured through ``hackerargs``:
  - ``args['model']``: backend name.
  - ``args['model.dir']``: directory containing model files.
  - ``args[f'model.<task>-<sd>']``: filename (LGBM) or stem-without-extension (transformer).

For transformer backends the suite reads a ``.meta`` JSON next to the
``.safetensors`` file. The meta is the **only** source of truth for
``input_dim``, ``d_model``, ``n_heads``, ``n_layers``, ``ffn_dim``,
``n_classes``. No more hardcoded dims.

Inference for transformer backends runs **2-pass autoregressive**:
  - Pass 1: prev_limb = START token for every position.
  - Pass 2: prev_limb taken from pass-1 predictions.

This matches the AR validation routine in ``train_mlx.py:evaluate_ar_accuracy``
and the production path in ``cli/limbuse/infer_v8_batch.py:ar_infer``.

Chunk size matches training (``MAX_SEQ_LEN = 1024``) with linear overlap
windowing so charts longer than the window are stitched seamlessly.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from hackerargs import args
from loguru import logger

import lightgbm as lgb
from lightgbm import Booster
from sklearn.model_selection import train_test_split


SUPPORTED_BACKENDS = ('lightgbm', 'mlx', 'torch')
TRANSFORMER_BACKENDS = ('mlx', 'torch')

MAX_SEQ_LEN = 1024
CHUNK_OVERLAP = 256


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    def predict(self, points: NDArray) -> NDArray:
        raise NotImplementedError

    def predict_prob(self, points: NDArray) -> NDArray:
        raise NotImplementedError

    def predict_log_prob(self, points: NDArray) -> NDArray:
        raise NotImplementedError


class DummyMatchModel(ModelWrapper):
    """Constant 0.5 probability. Used when match_next/match_prev aren't trained
    for a transformer backend. The tactician score term becomes a no-op (log 0.5
    is a constant offset, doesn't affect argmax)."""

    def predict(self, points: NDArray) -> NDArray:
        return np.zeros(len(points), dtype=int)

    def predict_prob(self, points: NDArray) -> NDArray:
        return np.full((len(points), 2), 0.5)

    def predict_log_prob(self, points: NDArray) -> NDArray:
        return np.full((len(points), 2), np.log(0.5))


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBModel(ModelWrapper):
    def __init__(self, bst: Booster):
        self.bst = bst

    @staticmethod
    def load(file: str) -> 'LGBModel':
        return LGBModel(lgb.Booster(model_file=file))

    @staticmethod
    def train(points: NDArray, labels: NDArray) -> 'LGBModel':
        train_x, test_x, train_y, test_y = train_test_split(points, labels)
        train_data = lgb.Dataset(train_x, label=train_y)
        test_data = lgb.Dataset(test_x, label=test_y)
        params = {'objective': 'binary', 'metric': 'binary_logloss'}
        bst = lgb.train(params, train_data, valid_sets=[test_data])
        return LGBModel(bst)

    def save(self, file: str) -> None:
        self.bst.save_model(file)

    def predict(self, points: NDArray) -> NDArray:
        return self.bst.predict(points).round().astype(int)

    def predict_prob(self, points: NDArray) -> NDArray:
        p = self.bst.predict(points)
        return np.stack([1 - p, p]).T

    def predict_log_prob(self, points: NDArray) -> NDArray:
        return np.log(np.maximum(self.predict_prob(points), 1e-10))


# ---------------------------------------------------------------------------
# Shared AR-inference helpers (numpy, backend-agnostic)
# ---------------------------------------------------------------------------

def _prev_limb_onehot_from_preds(preds: NDArray, first_label: int = 3) -> NDArray:
    """Build (N, 4) one-hot of previous label. ``first_label`` for position 0.
    Classes: L=0, R=1, E=2, START=3."""
    n = len(preds)
    prev = np.zeros((n, 4), dtype=np.float32)
    prev[0, min(int(first_label), 3)] = 1.0
    for i in range(1, n):
        lbl = min(int(preds[i - 1]), 2)
        prev[i, lbl] = 1.0
    return prev


def _all_start_prev_limb(n: int) -> NDArray:
    """Pass-1 prev_limb: START token at position 0, zeros elsewhere."""
    prev = np.zeros((n, 4), dtype=np.float32)
    if n > 0:
        prev[0, 3] = 1.0
    return prev


def _chunk_slices(n: int, max_len: int = MAX_SEQ_LEN, overlap: int = CHUNK_OVERLAP):
    """Yield (chunk_array, slice) pairs covering [0, n) with overlap."""
    if n <= max_len:
        yield slice(0, n)
        return
    stride = max_len - overlap
    start = 0
    while start + max_len <= n:
        yield slice(start, start + max_len)
        start += stride
    if start < n:
        yield slice(n - max_len, n)


def _ramp_weights(chunk_len: int, overlap: int = CHUNK_OVERLAP) -> NDArray:
    """Linear ramp at chunk borders to weight overlap regions."""
    w = np.ones(chunk_len, dtype=np.float64)
    if chunk_len > 2 * overlap:
        ramp = np.linspace(0.3, 1.0, overlap)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


# ---------------------------------------------------------------------------
# Transformer backends
# ---------------------------------------------------------------------------

class TransformerModelBase(ModelWrapper):
    """Shared AR + chunked inference plumbing. Subclasses implement
    ``_forward_logits(x_np: (L, D)) -> (L, C) np.ndarray``."""

    def __init__(self, meta: dict[str, Any]):
        self.meta = meta
        # ``meta['input_dim']`` follows the convention written by
        # ``train_mlx.py`` and ``train_torch.py``: it is the FULL transformer
        # input dim — including the 4 prev_limb one-hot dims. The numpy
        # ``points`` array passed to ``predict_prob`` has the *raw* shape
        # (features before prev_limb); ``_ar_logits`` appends the prev_limb
        # one-hot internally.
        self.full_input_dim = meta['input_dim']
        self.prev_dim = 4
        self.input_dim = self.full_input_dim - self.prev_dim
        self.n_classes = meta.get('n_classes', 3)

    # subclass contract
    def _forward_logits(self, x_chunk: NDArray) -> NDArray:
        raise NotImplementedError

    def _sequence_logits(self, x_full: NDArray) -> NDArray:
        """Run chunked forward over a full sequence; average overlaps."""
        n = len(x_full)
        if n == 0:
            return np.zeros((0, self.n_classes), dtype=np.float64)
        acc = np.zeros((n, self.n_classes), dtype=np.float64)
        w_acc = np.zeros(n, dtype=np.float64)
        for slc in _chunk_slices(n):
            chunk = x_full[slc]
            logits = self._forward_logits(chunk).astype(np.float64)
            w = _ramp_weights(len(chunk))
            acc[slc] += logits * w[:, None]
            w_acc[slc] += w
        return acc / np.maximum(w_acc, 1e-8)[:, None]

    def _ar_logits(self, points: NDArray) -> NDArray:
        """2-pass AR inference. ``points`` has shape (N, input_dim) — no prev_limb yet."""
        n = len(points)
        if n == 0:
            return np.zeros((0, self.n_classes), dtype=np.float64)
        # Pass 1
        prev1 = _all_start_prev_limb(n)
        x1 = np.concatenate([points.astype(np.float32), prev1], axis=1)
        logits1 = self._sequence_logits(x1)
        preds1 = np.argmax(logits1, axis=-1)
        # Pass 2
        prev2 = _prev_limb_onehot_from_preds(preds1, first_label=3)
        x2 = np.concatenate([points.astype(np.float32), prev2], axis=1)
        return self._sequence_logits(x2)

    def predict_prob(self, points: NDArray) -> NDArray:
        """Returns (N, n_classes) softmax probabilities (3-class L/R/E)."""
        logits = self._ar_logits(points)
        m = np.max(logits, axis=-1, keepdims=True)
        e = np.exp(logits - m)
        p = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)
        return p

    def predict(self, points: NDArray) -> NDArray:
        """Force L/R choice. ``Either`` (class 2) collapses to whichever has
        higher probability between L and R — never returned as a hard prediction."""
        p = self.predict_prob(points)
        if p.shape[-1] > 2:
            return np.argmax(p[:, :2], axis=-1)
        return np.argmax(p, axis=-1)

    def predict_log_prob(self, points: NDArray) -> NDArray:
        return np.log(np.maximum(self.predict_prob(points), 1e-10))


class MLXModel(TransformerModelBase):
    def __init__(self, meta: dict, mx_module, model):
        super().__init__(meta)
        self._mx = mx_module
        self.model = model

    @staticmethod
    def load(weights_path: str, meta: dict) -> 'MLXModel':
        import mlx.core as mx
        from piu_annotate.ml.mlx_architecture import LimbSequenceTransformer

        # meta['input_dim'] is the full model input dim (already includes prev_limb)
        model = LimbSequenceTransformer(
            input_dim=meta['input_dim'],
            d_model=meta.get('d_model', 256),
            n_heads=meta.get('n_heads', 8),
            n_layers=meta.get('n_layers', 6),
            ffn_dim=meta.get('ffn_dim', 1024),
            n_classes=meta.get('n_classes', 3),
        )
        model.load_weights(weights_path)
        mx.eval(model.parameters())
        model.eval()
        return MLXModel(meta, mx, model)

    def _forward_logits(self, x_chunk: NDArray) -> NDArray:
        mx = self._mx
        L = len(x_chunk)
        x_mx = mx.array(x_chunk[None].astype(np.float32))
        pm = mx.zeros((1, L), dtype=mx.bool_)
        logits = self.model(x_mx, pm)
        mx.eval(logits)
        return np.array(logits)[0]


class RefineModel(TransformerModelBase):
    """Two-pass coarse → refine wrapper.

    At predict time it runs the coarse model first, concatenates the coarse
    softmax to the raw features, and feeds the result through the refine
    model. The refine model's ``meta['input_dim']`` is ``raw_dim + 3``
    (coarse soft probs).

    Both ``coarse`` and ``refine`` are themselves ``TransformerModelBase``
    instances — so the same wrapper works for any combination of MLX and
    PyTorch backends, at the cost of crossing the numpy boundary between
    them (always returns to numpy after coarse, then re-enters the refine
    backend). For single-backend setups this is the canonical path.
    """

    def __init__(self, coarse: 'TransformerModelBase', refine: 'TransformerModelBase'):
        # Build meta from refine's view: raw input is coarse's raw input.
        meta = {
            'input_dim': coarse.input_dim,
            'n_classes': refine.n_classes,
            'is_refine': True,
        }
        super().__init__(meta)
        self.coarse = coarse
        self.refine = refine

    def _forward_logits(self, x_chunk):
        raise RuntimeError('RefineModel does not run a single forward — use predict_prob.')

    def predict_prob(self, points):
        coarse_p = self.coarse.predict_prob(points)  # (N, 3)
        extended = np.concatenate([points.astype(np.float32), coarse_p.astype(np.float32)], axis=1)
        return self.refine.predict_prob(extended)


class TorchModel(TransformerModelBase):
    def __init__(self, meta: dict, torch_module, model, device):
        super().__init__(meta)
        self._torch = torch_module
        self.model = model
        self.device = device

    @staticmethod
    def load(weights_path: str, meta: dict, device: str | None = None) -> 'TorchModel':
        from piu_annotate.ml.arch_torch import (
            build_model, pick_device, load_safetensors,
        )
        import torch

        dev = pick_device(prefer=device)
        # meta['input_dim'] is the full model input dim (already includes prev_limb)
        model = build_model(
            input_dim=meta['input_dim'],
            d_model=meta.get('d_model', 256),
            n_heads=meta.get('n_heads', 8),
            n_layers=meta.get('n_layers', 6),
            ffn_dim=meta.get('ffn_dim', 1024),
            n_classes=meta.get('n_classes', 3),
        )
        load_safetensors(model, weights_path, device=dev)
        model.eval()
        return TorchModel(meta, torch, model, dev)

    def _forward_logits(self, x_chunk: NDArray) -> NDArray:
        torch = self._torch
        L = len(x_chunk)
        with torch.inference_mode():
            x_t = torch.from_numpy(x_chunk[None].astype(np.float32)).to(self.device)
            pm = torch.zeros(1, L, dtype=torch.bool, device=self.device)
            logits = self.model(x_t, pm)
            return logits[0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

class ModelSuite:
    """Loads the four sub-models for a singles/doubles set.

    Public attributes used by the tactician:
      - ``model_arrows_to_limb``
      - ``model_arrowlimbs_to_limb``
      - ``model_arrows_to_matchnext``
      - ``model_arrows_to_matchprev``
      - ``model_type``: 'lightgbm' | 'mlx' | 'torch'
      - ``is_transformer``: bool
    """

    def __init__(self, singles_or_doubles: str, backend: str | None = None):
        self.singles_or_doubles = singles_or_doubles
        self.model_type = backend or args.setdefault('model', 'lightgbm')
        if self.model_type not in SUPPORTED_BACKENDS:
            raise ValueError(f'Unknown backend {self.model_type!r}; supported: {SUPPORTED_BACKENDS}')
        self.is_transformer = self.model_type in TRANSFORMER_BACKENDS

        sd = singles_or_doubles
        if self.model_type == 'lightgbm':
            self.model_arrows_to_limb       = self._load_lgbm(f'arrows_to_limb-{sd}')
            self.model_arrowlimbs_to_limb   = self._load_lgbm(f'arrowlimbs_to_limb-{sd}')
            self.model_arrows_to_matchnext  = self._load_lgbm(f'arrows_to_matchnext-{sd}')
            self.model_arrows_to_matchprev  = self._load_lgbm(f'arrows_to_matchprev-{sd}')
        else:
            coarse = self._load_transformer('arrows_to_limb', sd)
            self.model_arrows_to_limb = coarse
            # If a refine model exists, wrap (coarse, refine) as a RefineModel
            # exposing the same predict_prob API. Otherwise fall back to the
            # coarse model so the tactician's arrowlimbs path stays a no-op.
            refine_loaded = self._load_transformer(
                'arrowlimbs_to_limb', sd, fallback=None, optional=True,
            )
            if refine_loaded is None:
                self.model_arrowlimbs_to_limb = coarse
            else:
                meta = getattr(refine_loaded, 'meta', {}) or {}
                if meta.get('is_refine'):
                    self.model_arrowlimbs_to_limb = RefineModel(coarse, refine_loaded)
                else:
                    self.model_arrowlimbs_to_limb = refine_loaded
            # match models aren't trained for transformer backends → dummy
            self.model_arrows_to_matchnext  = DummyMatchModel()
            self.model_arrows_to_matchprev  = DummyMatchModel()

    # -- LGBM --

    def _load_lgbm(self, key: str) -> LGBModel:
        model_file = os.path.join(args['model.dir'], args[f'model.{key}'])
        return LGBModel.load(model_file)

    # -- Transformer --

    def _resolve_transformer_paths(self, task: str, sd: str) -> tuple[str, str]:
        """Return (weights_path, meta_path) for a given task+sd. Honors
        ``args['model.<task>-<sd>']`` as a stem override; otherwise uses the
        canonical name ``<sd>-<task>-<backend>-best.safetensors``."""
        cfg_key = f'model.{task}-{sd}'
        stem = args.get(cfg_key)
        if not stem:
            stem = f'{sd}-{task}-{self.model_type}-best'
        # tolerate user storing with or without ``.safetensors`` extension
        if stem.endswith('.safetensors'):
            stem = stem[:-len('.safetensors')]
        weights = os.path.join(args['model.dir'], f'{stem}.safetensors')
        meta = os.path.join(args['model.dir'], f'{stem}.meta')
        return weights, meta

    def _load_transformer(
        self,
        task: str,
        sd: str,
        fallback: str | None = None,
        optional: bool = False,
    ) -> ModelWrapper | None:
        weights, meta_path = self._resolve_transformer_paths(task, sd)
        if not os.path.exists(weights):
            if fallback:
                logger.warning(
                    f'{task}-{sd}: weights not found at {weights}, falling back to {fallback}-{sd}'
                )
                return self._load_transformer(fallback, sd, fallback=None)
            if optional:
                return None
            raise FileNotFoundError(f'No transformer weights at {weights}')

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            # Last-resort: read train_config.json in the same dir and fill canonical dims
            cfg_path = os.path.join(args['model.dir'], 'train_config.json')
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(
                    f'No .meta nor train_config.json next to {weights}. Cannot infer model dims.'
                )
            with open(cfg_path) as f:
                cfg = json.load(f)
            meta = {
                'input_dim': cfg['input_dim'],   # full dim including prev_limb
                'd_model':   cfg.get('d_model', 256),
                'n_heads':   cfg.get('n_heads', 8),
                'n_layers':  cfg.get('n_layers', 6),
                'ffn_dim':   cfg.get('ffn_dim', 1024),
                'n_classes': cfg.get('n_classes', 3),
            }
            logger.warning(f'{task}-{sd}: synthesized meta from train_config.json')

        if self.model_type == 'mlx':
            return MLXModel.load(weights, meta)
        elif self.model_type == 'torch':
            return TorchModel.load(weights, meta, device=args.get('model.device'))
        else:
            raise RuntimeError(f'Unreachable: model_type={self.model_type}')


# ---------------------------------------------------------------------------
# Inference cache (per-suite, per-input identity)
# ---------------------------------------------------------------------------
# The tactician calls ``predict_prob`` repeatedly during beam search. The
# transformer AR pass is expensive, so the model wrappers above already do
# their work once per (model, input). Use this lru_cache only if you need a
# global cache across suites.

@functools.lru_cache(maxsize=4)
def get_suite(singles_or_doubles: str, backend: str | None = None) -> ModelSuite:
    return ModelSuite(singles_or_doubles, backend=backend)
