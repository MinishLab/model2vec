from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import torch
from datasets import Column
from tokenizers import Tokenizer
from torch import nn

from model2vec.model import DEFAULT_MAX_LENGTH, PathLike, StaticModel
from model2vec.train.base import BaseFinetuneable, _load_static_model, _static_model_arguments
from model2vec.train.dataset import get_vector_dims_from_column
from model2vec.train.utils import DEFAULT_RANDOM_SEED, seed_everything

logger = logging.getLogger(__name__)


def _vector_dim(vectors: Any, name: str) -> int:
    """Get the dimension of a set of vectors, checking that every vector is present and has the same dimension.

    :param vectors: The vectors: a 2D tensor or array, a sequence of sequences of numbers, or a column of a Hugging
        Face dataset that holds lists of numbers.
    :param name: The name of the vectors, used in error messages.
    :return: The dimension of the vectors.
    :raises ValueError: If there are no vectors, if a vector is missing, if the vectors have different dimensions,
        or if a column doesn't hold lists of numbers.
    """
    if isinstance(vectors, (torch.Tensor, np.ndarray)):
        if vectors.ndim != 2:
            raise ValueError(f"{name} must be 2-dimensional, got {vectors.ndim} dimensions.")
        return vectors.shape[1]

    if isinstance(vectors, Column):
        dims = get_vector_dims_from_column(vectors, name)
    else:
        try:
            dims = {len(vector) for vector in vectors}
        except TypeError:
            raise ValueError(f"Vectors in {name} must be sequences of numbers.") from None

    if not dims:
        raise ValueError(f"{name} must not be empty.")
    if len(dims) > 1:
        raise ValueError(f"All vectors in {name} must have the same dimension, got {sorted(dims)}.")
    return dims.pop()


class CosineLoss(nn.Module):
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns the cosine distance loss function."""
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return (1 - torch.sum(x * y, dim=1)).mean()


class StaticModelForSimilarity(BaseFinetuneable):
    val_metric = "val_loss"
    early_stopping_direction = "min"

    @staticmethod
    def _build_loss_function() -> nn.Module:
        """Construct the loss function used to train this model."""
        return CosineLoss()

    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        n_layers: int = 1,
        hidden_dim: int = 512,
        out_dim: int = 2,
        pad_id: int = 0,
        token_mapping: list[int] | None = None,
        weights: torch.Tensor | None = None,
        freeze: bool = False,
        normalize: bool = True,
        freeze_weights: bool = False,
        max_length: int | None = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Initialize a standard similarity model."""
        super().__init__(
            vectors=vectors,
            out_dim=out_dim,
            pad_id=pad_id,
            tokenizer=tokenizer,
            token_mapping=token_mapping,
            weights=weights,
            freeze=freeze,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            normalize=normalize,
            freeze_weights=freeze_weights,
            max_length=max_length,
        )

    @classmethod
    def from_pretrained(
        cls: type[T],
        path: PathLike = "minishlab/potion-base-32m",
        *,
        token: str | None = None,
        model_name: PathLike | None = None,
        pad_token: str | None = None,
        max_length: int | None = None,
        n_layers: int = 1,
        hidden_dim: int = 512,
        out_dim: int = 2,
        freeze: bool = False,
        normalize: bool = True,
        freeze_weights: bool = False,
    ) -> T:
        """Load the model from a pretrained model2vec model.

        :param path: The path to the folder containing the model, or a repository on the Hugging Face Hub.
        :param token: The token to use to download the model from the hub.
        :param model_name: Deprecated alias for `path`.
        :param pad_token: The token to use for padding. If None, it is inferred from the tokenizer.
        :param max_length: The default maximum sequence length to use for tokenization. If None, the
            static model's `max_length` is used.
        :param n_layers: The number of hidden layers in the head.
        :param hidden_dim: The hidden dimension of the head.
        :param out_dim: The output dimension of the head. This is reset when calling `fit`.
        :param freeze: Whether to freeze the embeddings.
        :param normalize: Whether to normalize the embeddings.
        :param freeze_weights: Whether to freeze the learned token weights.
        :return: The initialized model.
        """
        model = _load_static_model(path, token=token, model_name=model_name)
        return cls.from_static_model(
            model=model,
            pad_token=pad_token,
            max_length=max_length,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            freeze=freeze,
            normalize=normalize,
            freeze_weights=freeze_weights,
        )

    @classmethod
    def from_static_model(
        cls: type[T],
        *,
        model: StaticModel,
        pad_token: str | None = None,
        max_length: int | None = None,
        n_layers: int = 1,
        hidden_dim: int = 512,
        out_dim: int = 2,
        freeze: bool = False,
        normalize: bool = True,
        freeze_weights: bool = False,
    ) -> T:
        """Load the model from a static model.

        :param model: The static model to load from.
        :param pad_token: The token to use for padding. If None, it is inferred from the tokenizer.
        :param max_length: The default maximum sequence length to use for tokenization. If None, the
            static model's `max_length` is used.
        :param n_layers: The number of hidden layers in the head.
        :param hidden_dim: The hidden dimension of the head.
        :param out_dim: The output dimension of the head. This is reset when calling `fit`.
        :param freeze: Whether to freeze the embeddings.
        :param normalize: Whether to normalize the embeddings.
        :param freeze_weights: Whether to freeze the learned token weights.
        :return: The initialized model.
        """
        return cls(
            **_static_model_arguments(model, pad_token=pad_token, max_length=max_length),
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            freeze=freeze,
            normalize=normalize,
            freeze_weights=freeze_weights,
        )

    def fit(
        self: T,
        X: Sequence[str],
        y: torch.Tensor | Sequence[Sequence[float]],
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float | int = 0.1,
        device: str = "auto",
        X_val: Sequence[str] | None = None,
        y_val: torch.Tensor | Sequence[Sequence[float]] | None = None,
        validation_steps: int | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        token_dropout: float = 0.0,
    ) -> T:
        """Fit a model.

        This function trains the model with a plain torch training loop.
        We use early stopping. After training, the weights of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        If `X_val` and `y_val` are not provided, the function will automatically
        split the training data into a train and validation set using `test_size`.

        The texts and vectors are read and tokenized per batch. They can be lists or tensors, or columns of a
        Hugging Face dataset, such as `dataset["text"]`, which are not loaded into memory. The dataset must not have a
        transform.

        :param X: The texts to train on.
        :param y: The vectors to train on.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size. If None, a good batch size is chosen automatically.
        :param min_epochs: The minimum number of epochs to train for.
        :param max_epochs: The maximum number of epochs to train for.
            If this is -1, the model trains until early stopping is triggered.
        :param early_stopping_patience: The patience for early stopping.
            If this is None, early stopping is disabled.
        :param test_size: The size of the validation split if `X_val` is None: a fraction of the data, capped at
            10,000 rows, or a number of rows if it is an int.
        :param device: The device to train on. If this is "auto", the device is chosen automatically.
        :param X_val: The texts to be used for validation.
        :param y_val: The vectors to be used for validation.
        :param validation_steps: The number of steps to run validation for. If None, validation steps are estimated from the data.
        :param random_seed: The random seed to use. Defaults to 42.
        :param token_dropout: The fraction of tokens to randomly drop from each training sample.
            Has no effect during validation. Must be in the range [0, 1).
        :return: The fitted model.
        :raises ValueError: If the vectors in `y_val` have a different dimension than those in `y`.
        """
        seed_everything(random_seed)
        logger.info("Re-initializing model.")
        self._check_inputs(X=X, y=y, X_val=X_val, y_val=y_val)
        out_dim = _vector_dim(y, "y")
        if y_val is not None and (val_dim := _vector_dim(y_val, "y_val")) != out_dim:
            raise ValueError(f"The vectors in y_val have dimension {val_dim}, but those in y have dimension {out_dim}.")

        train_dataset, val_dataset = self._create_datasets(X, y, X_val, y_val, test_size)
        self.out_dim = out_dim
        self._initialize()
        self._train(
            loss_function=self._build_loss_function(),
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=self._determine_batch_size(batch_size, len(train_dataset)),
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            device=device,
            validation_steps=validation_steps,
            token_dropout=token_dropout,
        )

        return self


T = TypeVar("T", bound=StaticModelForSimilarity)
