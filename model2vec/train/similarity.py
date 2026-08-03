from __future__ import annotations

import logging
from typing import TypeVar

import torch
from tokenizers import Tokenizer
from torch import nn

from model2vec.train.base import BaseFinetuneable
from model2vec.train.utils import DEFAULT_RANDOM_SEED, seed_everything

logger = logging.getLogger(__name__)


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
        )

    def fit(
        self: _T,
        X: list[str],
        y: torch.Tensor,
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
        X_val: list[str] | None = None,
        y_val: torch.Tensor | None = None,
        validation_steps: int | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> _T:
        """Fit a model.

        This function trains the model with a plain torch training loop.
        We use early stopping. After training, the weights of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        If `X_val` and `y_val` are not provided, the function will automatically
        split the training data into a train and validation set using `test_size`.

        :param X: The texts to train on.
        :param y: The vectors to train on.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size. If None, a good batch size is chosen automatically.
        :param min_epochs: The minimum number of epochs to train for.
        :param max_epochs: The maximum number of epochs to train for.
            If this is -1, the model trains until early stopping is triggered.
        :param early_stopping_patience: The patience for early stopping.
            If this is None, early stopping is disabled.
        :param test_size: The test size for the train-test split.
        :param device: The device to train on. If this is "auto", the device is chosen automatically.
        :param X_val: The texts to be used for validation.
        :param y_val: The vectors to be used for validation.
        :param validation_steps: The number of steps to run validation for. If None, validation steps are estimated from the data.
        :param random_seed: The random seed to use. Defaults to 42.
        :return: The fitted model.
        """
        seed_everything(random_seed)
        logger.info("Re-initializing model.")

        train_dataset, val_dataset = self._create_datasets(X, y, X_val, y_val, test_size)
        batch_size = self._determine_batch_size(batch_size, len(train_dataset))

        self.out_dim = train_dataset.targets.shape[1]
        self._initialize()

        self._train(
            loss_function=self._build_loss_function(),
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            device=device,
            validation_steps=validation_steps,
        )

        return self


_T = TypeVar("_T", bound=StaticModelForSimilarity)
