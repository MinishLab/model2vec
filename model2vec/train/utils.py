from __future__ import annotations

import logging
import numbers
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation, Layer, MLPHead
from model2vec.train.dataset import stratify_indices

if TYPE_CHECKING:
    from model2vec.train.base import BaseFinetuneable
    from model2vec.train.classifier import StaticModelForClassification


logger = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 42
MAX_VALIDATION_SIZE = 10_000


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


def _stratum_test_sizes(sizes: np.ndarray, n_test: int) -> np.ndarray:
    """Divide `n_test` test items over strata in proportion to their sizes.

    Every stratum gets at least one test item and keeps at least one train item. The total only differs from
    `n_test` if this can't be met otherwise.

    :param sizes: The number of items in each stratum. Each stratum has at least two items.
    :param n_test: The total number of test items.
    :return: The number of test items in each stratum.
    """
    quotas = sizes * n_test / sizes.sum()
    counts = np.clip(np.round(quotas), 1, sizes - 1).astype(int)
    while (excess := int(counts.sum()) - n_test) != 0:
        step = 1 if excess > 0 else -1
        candidates = np.flatnonzero(counts > 1 if excess > 0 else counts < sizes - 1)
        if not len(candidates):
            break
        index = candidates[np.argmax((counts - quotas)[candidates] * step)]
        counts[index] -= step
    return counts


def split_indices(
    n: int,
    test_size: float | int,
    max_test_size: int | None = None,
    stratify_by: Sequence[Any] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split the indices `0..n-1` into sorted train and test indices.

    :param n: The number of items.
    :param test_size: The size of the test split: a fraction of the items if it is a float, or a number of items
        if it is an int. At least one item goes into the test split, and at least one into the train split if
        `n > 1`.
    :param max_test_size: The maximum number of items in the test split if `test_size` is a fraction.
        If None, the test split is not capped.
    :param stratify_by: The single label of each item, as strings or integers that have been validated. If every
        label occurs at least twice, each label is split separately, in the same proportion. If None, the split is
        not stratified.
    :param random_seed: The random seed of the split.
    :return: The train indices and the test indices.
    :raises ValueError: If `test_size` is a bool.
    """
    if isinstance(test_size, bool):
        raise ValueError("test_size must be a float or an int, not a bool.")
    rng = np.random.default_rng(random_seed)
    if isinstance(test_size, numbers.Integral):
        n_test = int(test_size)
    else:
        n_test = round(n * test_size)
        if max_test_size is not None:
            n_test = min(n_test, max_test_size)
    n_test = min(max(1, n_test), max(n - 1, 0))

    strata = None if stratify_by is None else stratify_indices(stratify_by)
    if strata is not None and len(strata) > n_test:
        logger.info("There are more classes than validation samples. Stratification is disabled.")
        strata = None
    if strata is None:
        indices = rng.permutation(n)
        return np.sort(indices[n_test:]), np.sort(indices[:n_test])

    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    test_sizes = _stratum_test_sizes(np.array([len(members) for members in strata]), n_test)
    for members, n_members_test in zip(strata, test_sizes):
        members = rng.permutation(members)
        test.append(members[:n_members_test])
        train.append(members[n_members_test:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch random number generators."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
