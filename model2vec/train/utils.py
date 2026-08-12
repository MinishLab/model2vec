from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn

from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation, Layer, MLPHead

if TYPE_CHECKING:
    from model2vec.train.base import BaseFinetuneable
    from model2vec.train.classifier import StaticModelForClassification


logger = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 42
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


def to_pipeline(model: "BaseFinetuneable | StaticModelForClassification") -> StaticModelPipeline:
    """Convert the model to an inference pipeline."""
    from model2vec.train.classifier import StaticModelForClassification

    static_model = model.to_static_model()

    layers = [
        Layer(weight=module.weight.detach().cpu().numpy(), bias=module.bias.detach().cpu().numpy())
        for module in model.head
        if isinstance(module, nn.Linear)
    ]

    classes: np.ndarray | None = None
    if isinstance(model, StaticModelForClassification):
        classes = np.asarray(model.classes_)
        activation = Activation.SIGMOID if model.multilabel else Activation.SOFTMAX
    else:
        activation = Activation.IDENTITY

    head = MLPHead(layers=layers, activation=activation, classes=classes)

    return StaticModelPipeline(static_model, head)


def _index(sequence: Any, indices: list[int]) -> Any:
    """Index a sequence with a list of indices, preserving the container type."""
    if isinstance(sequence, list):
        return [sequence[index] for index in indices]
    return sequence[indices]


def train_test_split(
    X: list[str],
    y: list,
    test_size: float,
) -> tuple[list[str], list[str], list, list]:
    """Split the data.

    For single-label classification, stratification is attempted (if possible).
    For multilabel classification, a random split is performed.
    """
    rng = random.Random(DEFAULT_RANDOM_SEED)
    n = len(X)

    stratify = isinstance(y, list) and len(y) > 0 and isinstance(y[0], (str, int))
    if stratify:
        label_counts = Counter(y)
        if min(label_counts.values()) < 2:
            logger.info("Some classes have fewer than 2 samples. Stratification is disabled.")
            stratify = False

    train_indices: list[int]
    test_indices: list[int]
    if stratify:
        indices_by_label: dict[Any, list[int]] = defaultdict(list)
        for index, label in enumerate(y):
            indices_by_label[label].append(index)

        train_indices = []
        test_indices = []
        for indices in indices_by_label.values():
            indices = indices[:]
            rng.shuffle(indices)
            n_test = min(max(1, round(len(indices) * test_size)), len(indices) - 1)
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    else:
        indices = list(range(n))
        rng.shuffle(indices)
        n_test = min(max(1, round(n * test_size)), max(n - 1, 0))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    X_train = _index(X, train_indices)
    X_test = _index(X, test_indices)
    y_train = _index(y, train_indices)
    y_test = _index(y, test_indices)

    return X_train, X_test, y_train, y_test


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch random number generators."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logit(x: torch.Tensor) -> torch.Tensor:
    """Invert a sigmoid."""
    return -torch.log((1 / x) - 1)
