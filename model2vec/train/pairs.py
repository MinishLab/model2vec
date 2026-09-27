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

        Every first text labeled 1 is treated as an anchor, its paired second text as the positive, and
        all other second texts in the batch as negatives. Pairs labeled 0 are not used as anchors, but
        their second texts still serve as negatives for the other anchors. A second text is not used as a
        negative for an anchor if it is identical to the anchor's positive, or if it belongs to another
        pair labeled 1 with an identical first text.

        :param head_out: The encoded first texts and second texts, followed by an id for each first text and
            each second text. Identical texts have the same id.
        :param y: The label of each pair.
        :return: The mean loss over the anchors.
        """
        out_a, out_b, ids_a, ids_b = head_out
        positive = y == 1
        out_a = torch.nn.functional.normalize(out_a, dim=1)
        out_b = torch.nn.functional.normalize(out_b, dim=1)
        logits = (out_a @ out_b.T) / self.temperature
        targets = torch.arange(len(logits), device=logits.device)
        same_positive = ids_b[:, None] == ids_b[None, :]
        other_positive = (ids_a[:, None] == ids_a[None, :]) & positive[None, :]
        false_negatives = same_positive | other_positive
        false_negatives[targets, targets] = False
        logits = logits.masked_fill(false_negatives, float("-inf"))
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        if not positive.any():
            return (loss * 0).sum()
        return loss[positive].mean()


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
        :param n_layers: The number of layers in the head.
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
        labels: list[int],
        text_a_val: list[str] | None,
        text_b_val: list[str] | None,
        labels_val: list[int] | None,
        test_size: float,
    ) -> tuple[list[str], list[str], list[str], list[str], list[int], list[int]]:
        if len(text_a) != len(text_b):
            raise ValueError("text_a and text_b must have the same length.")
        if len(labels) != len(text_a):
            raise ValueError("labels must have the same length as text_a and text_b.")
        if (text_a_val is not None) != (text_b_val is not None):
            raise ValueError("Both text_a_val and text_b_val must be provided together, or neither.")

        if text_a_val is not None and text_b_val is not None:
            if len(text_a_val) != len(text_b_val):
                raise ValueError("text_a_val and text_b_val must have the same length.")
            labels_val = [1] * len(text_a_val) if labels_val is None else labels_val
            if len(labels_val) != len(text_a_val):
                raise ValueError("labels_val must have the same length as text_a_val and text_b_val.")
            return text_a, text_a_val, text_b, text_b_val, labels, labels_val

        pairs = list(zip(text_a, text_b))
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, labels, test_size=test_size)
        train_a, train_b = map(list, zip(*train_pairs)) if train_pairs else ([], [])
        val_a, val_b = map(list, zip(*val_pairs)) if val_pairs else ([], [])
        return train_a, val_a, train_b, val_b, train_labels, val_labels

    @staticmethod
    def _check_pair_splits(train_labels: list[int], val_labels: list[int]) -> None:
        """Check that the training and validation sets each have at least two pairs, and a pair labeled 1.

        :param train_labels: The labels of the training pairs.
        :param val_labels: The labels of the validation pairs.
        :raises ValueError: If either set has fewer than two pairs, or no pair labeled 1.
        """
        for name, split_labels in (("training", train_labels), ("validation", val_labels)):
            if len(split_labels) < 2:
                raise ValueError(
                    f"The {name} set needs at least two pairs, got {len(split_labels)}. Pass more pairs, "
                    "a different test_size, or an explicit validation set."
                )
            if not any(label == 1 for label in split_labels):
                raise ValueError(f"The {name} set needs at least one pair labeled 1.")

    def _prepare_pair_dataset(
        self, text_a: list[str], text_b: list[str], labels: list[int], max_length: int | None
    ) -> PairDataset:
        """Tokenize both halves of a pair dataset.

        :param text_a: The first half of each pair.
        :param text_b: The second half of each pair.
        :param labels: The label for each pair.
        :param max_length: The maximum length of the input in tokens. If this is None, no truncation is done.
        :return: A PairDataset.
        """
        return PairDataset(
            self._tokenize_texts(text_a, max_length),
            self._tokenize_texts(text_b, max_length),
            labels=labels,
            pad_id=self.pad_id,
        )

    def fit(
        self: T,
        text_a: list[str],
        text_b: list[str],
        labels: list[int] | None = None,
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
        text_a_val: list[str] | None = None,
        text_b_val: list[str] | None = None,
        labels_val: list[int] | None = None,
        validation_steps: int | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        temperature: float = 0.05,
    ) -> T:
        """Fit a model that embeds paired texts close together.

        This function trains the model with a plain torch training loop. Both `text_a` and `text_b`
        are encoded with the same model, and trained with an InfoNCE loss: each `text_a` labeled 1 is
        pulled towards its paired `text_b` and pushed away from all other `text_b` in the batch. Pairs
        labeled 0 are not used as anchors, but their `text_b` still serves as an in-batch negative. We use
        early stopping. After training, the weights of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        If `text_a_val` and `text_b_val` are not provided, the function will automatically
        split the training data into a train and validation set using `test_size`.

        :param text_a: The first half of each training pair.
        :param text_b: The second half of each training pair.
        :param labels: The label for each training pair: 1 if the pair should be pushed together, 0 otherwise.
            If None, every pair is labeled 1.
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
        :param labels_val: The label for each validation pair. If None, every validation pair is labeled 1.
        :param validation_steps: The number of steps to run validation for. If None, validation steps are estimated from the data.
        :param random_seed: The random seed to use. Defaults to 42.
        :param temperature: The temperature of the InfoNCE loss.
        :return: The fitted model.
        :raises ValueError: If `batch_size` is smaller than 2.
        """
        seed_everything(random_seed)
        logger.info("Re-initializing model.")
        loss_function = PairInfoNCELoss(temperature=temperature)

        labels = [1] * len(text_a) if labels is None else labels

        train_a, val_a, train_b, val_b, train_labels, val_labels = self._check_pair_val_split(
            text_a, text_b, labels, text_a_val, text_b_val, labels_val, test_size
        )
        self._check_pair_splits(train_labels, val_labels)
        self._initialize()

        logger.info("Preparing train dataset.")
        train_dataset = self._prepare_pair_dataset(train_a, train_b, train_labels, self.max_length)
        logger.info("Preparing validation dataset.")
        val_dataset = self._prepare_pair_dataset(val_a, val_b, val_labels, self.max_length)

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
