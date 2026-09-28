from __future__ import annotations

import logging
from typing import TypeVar

import torch
from tokenizers import Tokenizer
from torch import nn

from model2vec.model import DEFAULT_MAX_LENGTH
from model2vec.train.base import BaseFinetuneable
from model2vec.train.dataset import PairDataset
from model2vec.train.utils import DEFAULT_RANDOM_SEED, seed_everything, train_test_split

logger = logging.getLogger(__name__)


class PairInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05) -> None:
        """Initialize the InfoNCE loss.

        :param temperature: The temperature by which the cosine similarities are divided. Must be positive.
        :raises ValueError: If `temperature` is not positive.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.temperature = temperature

    def __call__(
        self, head_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], y: torch.Tensor
    ) -> torch.Tensor:
        """Returns the InfoNCE loss over a pair batch, using in-batch negatives.

        Every first text is an anchor, its paired second text the positive, and all other second texts in
        the batch negatives. A second text is not used as a negative for an anchor if it is identical to
        the anchor's positive, or if it is paired with a first text identical to the anchor.

        :param head_out: The encoded first texts and second texts, followed by an id for each first text and
            each second text. Identical texts have the same id.
        :param y: For each anchor, the index of its positive among the second texts.
        :return: The mean loss over the anchors.
        """
        out_a, out_b, ids_a, ids_b = head_out
        out_a = torch.nn.functional.normalize(out_a, dim=1)
        out_b = torch.nn.functional.normalize(out_b, dim=1)
        logits = (out_a @ out_b.T) / self.temperature
        false_negatives = (ids_a[:, None] == ids_a[None, :]) | (ids_b[:, None] == ids_b[None, :])
        false_negatives[torch.arange(len(y), device=y.device), y] = False
        logits = logits.masked_fill(false_negatives, float("-inf"))
        return torch.nn.functional.cross_entropy(logits, y)


class StaticModelForPairSimilarity(BaseFinetuneable):
    val_metric = "val_loss"
    early_stopping_direction = "min"

    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        n_layers: int = 1,
        hidden_dim: int = 512,
        out_dim: int | None = None,
        pad_id: int = 0,
        token_mapping: list[int] | None = None,
        weights: torch.Tensor | None = None,
        freeze: bool = False,
        normalize: bool = True,
        freeze_weights: bool = False,
        max_length: int | None = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Initialize a model that is trained to embed pairs of texts close together.

        :param vectors: The embeddings of the staticmodel.
        :param tokenizer: The tokenizer.
        :param n_layers: The number of layers in the head. If this is 0 and `out_dim` equals the embedding
            dimension, the model has no head, and the embeddings are used as is.
        :param hidden_dim: The hidden dimension of the head.
        :param out_dim: The output embedding dimension. If None, defaults to the input embedding dimension.
        :param pad_id: The padding id. This is set to 0 in almost all model2vec models.
        :param token_mapping: The token mapping. If None, the token mapping is set to the range of the number of vectors.
        :param weights: The weights of the model. If None, the weights are initialized to zeros.
        :param freeze: Whether to freeze the embeddings. This should be set to False in most cases.
        :param normalize: Whether to normalize the embeddings.
        :param freeze_weights: Whether to freeze the learned token weights.
        :param max_length: The default maximum sequence length (in tokens) used to tokenize inputs.
        """
        super().__init__(
            vectors=vectors,
            out_dim=out_dim if out_dim is not None else vectors.shape[1],
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

    def forward(  # type: ignore[override]
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode both halves of a pair batch through the shared embeddings and head.

        :param input_ids: A `(2, batch_size, seq_len)` tensor, stacking the two padded text sets.
        :return: The head outputs for the first and second set of texts, followed by an id for each first
            text and each second text. Identical texts have the same id.
        """
        out_a = self.head(self._encode(input_ids[0]))
        out_b = self.head(self._encode(input_ids[1]))
        ids_a = torch.unique(input_ids[0], dim=0, return_inverse=True)[1]
        ids_b = torch.unique(input_ids[1], dim=0, return_inverse=True)[1]
        return out_a, out_b, ids_a, ids_b

    def _check_pair_val_split(
        self,
        text_a: list[str],
        text_b: list[str],
        text_a_val: list[str] | None,
        text_b_val: list[str] | None,
        test_size: float,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        if len(text_a) != len(text_b):
            raise ValueError("text_a and text_b must have the same length.")
        if (text_a_val is not None) != (text_b_val is not None):
            raise ValueError("Both text_a_val and text_b_val must be provided together, or neither.")

        if text_a_val is not None and text_b_val is not None:
            if len(text_a_val) != len(text_b_val):
                raise ValueError("text_a_val and text_b_val must have the same length.")
            return text_a, text_a_val, text_b, text_b_val

        pairs = list(zip(text_a, text_b))
        train_pairs, val_pairs, _, _ = train_test_split(pairs, pairs, test_size=test_size)
        train_a, train_b = map(list, zip(*train_pairs)) if train_pairs else ([], [])
        val_a, val_b = map(list, zip(*val_pairs)) if val_pairs else ([], [])
        return train_a, val_a, train_b, val_b

    @staticmethod
    def _check_pair_splits(n_train: int, n_val: int) -> None:
        """Check that the training and validation sets each have at least two pairs.

        :param n_train: The number of training pairs.
        :param n_val: The number of validation pairs.
        :raises ValueError: If either set has fewer than two pairs.
        """
        for name, n_pairs in (("training", n_train), ("validation", n_val)):
            if n_pairs < 2:
                raise ValueError(
                    f"The {name} set needs at least two pairs, got {n_pairs}. Pass more pairs, "
                    "a different test_size, or an explicit validation set."
                )

    def _prepare_pair_dataset(self, text_a: list[str], text_b: list[str], max_length: int | None) -> PairDataset:
        """Tokenize both halves of a pair dataset.

        :param text_a: The first half of each pair.
        :param text_b: The second half of each pair.
        :param max_length: The maximum length of the input in tokens. If this is None, no truncation is done.
        :return: A PairDataset.
        """
        return PairDataset(
            self._tokenize_texts(text_a, max_length),
            self._tokenize_texts(text_b, max_length),
            pad_id=self.pad_id,
        )

    def fit(
        self: T,
        text_a: list[str],
        text_b: list[str],
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
        text_a_val: list[str] | None = None,
        text_b_val: list[str] | None = None,
        validation_steps: int | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        temperature: float = 0.05,
    ) -> T:
        """Fit a model that embeds paired texts close together.

        This function trains the model with a plain torch training loop. Both `text_a` and `text_b`
        are encoded with the same model, and trained with an InfoNCE loss: each `text_a` is pulled towards
        its paired `text_b` and pushed away from all other `text_b` in the batch. Pairs with the same
        `text_a` are not used as negatives for each other. We use early stopping. After training, the weights of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        If `text_a_val` and `text_b_val` are not provided, the function will automatically
        split the training data into a train and validation set using `test_size`.

        :param text_a: The first half of each training pair.
        :param text_b: The second half of each training pair.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size. If None, a good batch size is chosen automatically.
        :param min_epochs: The minimum number of epochs to train for.
        :param max_epochs: The maximum number of epochs to train for.
            If this is -1, the model trains until early stopping is triggered.
        :param early_stopping_patience: The patience for early stopping.
            If this is None, early stopping is disabled.
        :param test_size: The test size for the train-test split.
        :param device: The device to train on. If this is "auto", the device is chosen automatically.
        :param text_a_val: The first half of each validation pair.
        :param text_b_val: The second half of each validation pair.
        :param validation_steps: The number of steps to run validation for. If None, validation steps are estimated from the data.
        :param random_seed: The random seed to use. Defaults to 42.
        :param temperature: The temperature of the InfoNCE loss.
        :return: The fitted model.
        :raises ValueError: If `batch_size` is smaller than 2.
        """
        seed_everything(random_seed)
        logger.info("Re-initializing model.")
        loss_function = PairInfoNCELoss(temperature=temperature)

        train_a, val_a, train_b, val_b = self._check_pair_val_split(text_a, text_b, text_a_val, text_b_val, test_size)
        self._check_pair_splits(len(train_a), len(val_a))
        self._initialize()

        logger.info("Preparing train dataset.")
        train_dataset = self._prepare_pair_dataset(train_a, train_b, self.max_length)
        logger.info("Preparing validation dataset.")
        val_dataset = self._prepare_pair_dataset(val_a, val_b, self.max_length)

        batch_size = self._determine_batch_size(batch_size, len(train_dataset))
        if batch_size < 2:
            raise ValueError(f"batch_size must be at least 2, got {batch_size}.")

        self._train(
            loss_function=loss_function,
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


T = TypeVar("T", bound=StaticModelForPairSimilarity)
