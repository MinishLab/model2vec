from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from tokenizers import Tokenizer
from torch import nn

from model2vec.inference import StaticModelPipeline

if TYPE_CHECKING:
    from model2vec.train.base import _BaseFinetuneable
    from model2vec.train.classifier import StaticModelForClassification


logger = logging.getLogger(__name__)

_KNOWN_PAD_TOKENS = ("[PAD]", "<pad>")


def get_probable_pad_token_id(tokenizer: Tokenizer) -> int:
    """Get a probable pad token by using the padding module and falling back to guessing."""
    if tokenizer.padding is not None:
        return tokenizer.padding["pad_id"]
    vocab = tokenizer.get_vocab()
    for token in _KNOWN_PAD_TOKENS:
        token_id = vocab.get(token)
        if token_id is not None:
            return token_id

    logger.warning("No known pad token found, using 0 as default")
    return 0


def to_pipeline(model: "_BaseFinetuneable | StaticModelForClassification") -> StaticModelPipeline:
    """Convert the model to an sklearn pipeline."""
    from model2vec.train.classifier import StaticModelForClassification

    static_model = model.to_static_model()

    random_state = np.random.RandomState(42)
    n_items = model.out_dim
    X = random_state.randn(n_items, static_model.dim)
    y: np.ndarray | list[str]
    if isinstance(model, StaticModelForClassification):
        y = model.classes_
        mlp_head = MLPClassifier(hidden_layer_sizes=(model.hidden_dim,) * model.n_layers)
        activation = "logistic" if model.multilabel else "softmax"
    else:
        y = random_state.randn(n_items, n_items)
        mlp_head = MLPRegressor(hidden_layer_sizes=(model.hidden_dim,) * model.n_layers)
        activation = "identity"
    mlp_head.fit(X, y)

    for index, layer in enumerate([module for module in model.head if isinstance(module, nn.Linear)]):
        mlp_head.coefs_[index] = layer.weight.detach().cpu().numpy().T
        mlp_head.intercepts_[index] = layer.bias.detach().cpu().numpy()

    mlp_head.n_outputs_ = model.out_dim
    mlp_head.out_activation_ = activation
    pipeline = make_pipeline(mlp_head)

    return StaticModelPipeline(static_model, pipeline)


def train_test_split(
    X: list[str],
    y: list,
    test_size: float,
) -> tuple[list[str], list[str], list, list]:
    """
    Split the data.

    For single-label classification, stratification is attempted (if possible).
    For multilabel classification, a random split is performed.
    """
    stratify_data = None
    if isinstance(y, list) and isinstance(y[0], (str, int)):
        label_counts = Counter(y)
        if min(label_counts.values()) < 2:
            logger.info("Some classes have fewer than 2 samples. Stratification is disabled.")
            stratify_data = None
        else:
            stratify_data = y
    return sklearn_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=stratify_data)  # type: ignore
