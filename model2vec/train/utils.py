from __future__ import annotations

import logging
import numbers
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Column
from tokenizers import Tokenizer
from torch import nn

from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation, Layer, MLPHead
from model2vec.train.dataset import column_type, iter_column

if TYPE_CHECKING:
    from model2vec.train.base import BaseFinetuneable
    from model2vec.train.classifier import StaticModelForClassification


logger = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 42
MAX_VALIDATION_SIZE = 10_000
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


def _list_strata(labels: list[Any]) -> list[np.ndarray]:
    """Group the indices of a list of labels by label, in order of first occurrence."""
    indices_by_label: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indices_by_label[label].append(index)
    return [np.asarray(indices) for indices in indices_by_label.values()]


def _array_strata(labels: np.ndarray) -> list[np.ndarray]:
    """Group the indices of an array of labels by label, in order of first occurrence."""
    _, first, codes = np.unique(labels, return_index=True, return_inverse=True)
    codes = np.argsort(np.argsort(first))[codes]
    return np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])


def _column_strata(labels: Column) -> list[np.ndarray] | None:
    """Group the indices of a Hugging Face column of single labels by label, in order of first occurrence.

    :param labels: The labels.
    :return: The indices of each label, or None if the column is empty or doesn't hold strings or integers.
    """
    label_type = column_type(labels)
    if not len(labels) or not (
        pa.types.is_string(label_type) or pa.types.is_large_string(label_type) or pa.types.is_integer(label_type)
    ):
        return None
    classes: dict[Any, int] = {}
    batch_codes = []
    for array in iter_column(labels):
        encoded = pc.dictionary_encode(array.combine_chunks(), null_encoding="encode")
        mapping = np.array([classes.setdefault(label, len(classes)) for label in encoded.dictionary.to_pylist()])
        batch_codes.append(mapping[encoded.indices.to_numpy(zero_copy_only=False)])
    codes = np.concatenate(batch_codes)
    return np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])


def _strata(labels: Sequence[Any] | None) -> list[np.ndarray] | None:
    """Group the indices of single-label classes, or return None if the labels can't be used to stratify."""
    if isinstance(labels, Column):
        strata = _column_strata(labels)
    elif isinstance(labels, torch.Tensor) and labels.ndim == 1 and len(labels):
        strata = _array_strata(labels.cpu().numpy())
    elif isinstance(labels, np.ndarray) and labels.ndim == 1 and len(labels):
        strata = _array_strata(labels)
    elif isinstance(labels, list) and labels and isinstance(labels[0], (str, int)):
        strata = _list_strata(labels)
    else:
        return None
    if strata is None:
        return None
    if min(len(indices) for indices in strata) < 2:
        logger.info("Some classes have fewer than 2 samples. Stratification is disabled.")
        return None
    return strata


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
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split the indices `0..n-1` into sorted train and test indices.

    :param n: The number of items.
    :param test_size: The size of the test split: a fraction of the items if it is a float, or a number of items
        if it is an int. At least one item goes into the test split, and at least one into the train split if
        `n > 1`.
    :param max_test_size: The maximum number of items in the test split if `test_size` is a fraction.
        If None, the test split is not capped.
    :param stratify_by: The label of each item. If this is a list or a Hugging Face column of single labels in which
        every label occurs at least twice, each label is split separately, in the same proportion.
    :return: The train indices and the test indices.
    :raises ValueError: If `test_size` is a bool.
    """
    if isinstance(test_size, bool):
        raise ValueError("test_size must be a float or an int, not a bool.")
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    if isinstance(test_size, numbers.Integral):
        n_test = int(test_size)
    else:
        n_test = round(n * test_size)
        if max_test_size is not None:
            n_test = min(n_test, max_test_size)
    n_test = min(max(1, n_test), max(n - 1, 0))

    strata = _strata(stratify_by)
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
