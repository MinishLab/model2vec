from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn

from model2vec.model import DEFAULT_MAX_LENGTH, PathLike, StaticModel
from model2vec.train.base import BaseFinetuneable, _load_static_model, _static_model_arguments
from model2vec.train.dataset import ColumnRows, PairDataset
from model2vec.train.utils import DEFAULT_RANDOM_SEED, MAX_VALIDATION_SIZE, seed_everything, split_indices

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
        out_dim: int | None = None,
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
        :param n_layers: The number of layers in the head. If this is 0 and `out_dim` equals the embedding
            dimension, the model has no head, and the embeddings are used as is.
        :param hidden_dim: The hidden dimension of the head.
        :param out_dim: The output embedding dimension. If None, defaults to the input embedding dimension.
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
        out_dim: int | None = None,
        freeze: bool = False,
        normalize: bool = True,
        freeze_weights: bool = False,
    ) -> T:
        """Load the model from a static model.

        :param model: The static model to load from.
        :param pad_token: The token to use for padding. If None, it is inferred from the tokenizer.
        :param max_length: The default maximum sequence length to use for tokenization. If None, the
            static model's `max_length` is used.
        :param n_layers: The number of layers in the head. If this is 0 and `out_dim` equals the embedding
            dimension, the model has no head, and the embeddings are used as is.
        :param hidden_dim: The hidden dimension of the head.
        :param out_dim: The output embedding dimension. If None, defaults to the input embedding dimension.
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

    def _pair_dataset(self, rows: ColumnRows, indices: np.ndarray | None = None) -> PairDataset:
        """Create a dataset of pairs that are tokenized per batch.

        :param rows: The pairs, in a `text_a` and a `text_b` column.
        :param indices: The indices of the rows that belong to the dataset. If None, all rows belong to it.
        :return: The dataset.
        """
        return PairDataset(rows, self._tokenize_ids, indices, pad_id=self.pad_id)

    def _create_pair_datasets(
        self,
        text_a: Sequence[str],
        text_b: Sequence[str],
        text_a_val: Sequence[str] | None,
        text_b_val: Sequence[str] | None,
        test_size: float | int,
    ) -> tuple[PairDataset, PairDataset]:
        """Create the training and validation datasets of pairs.

        :param text_a: The first half of each training pair.
        :param text_b: The second half of each training pair.
        :param text_a_val: The first half of each validation pair. If None, the validation pairs are split off
            from the training pairs.
        :param text_b_val: The second half of each validation pair.
        :param test_size: The size of the validation split if `text_a_val` is None: a fraction of the pairs,
            capped at `MAX_VALIDATION_SIZE` rows, or a number of pairs if it is an int.
        :return: The train and validation datasets.
        :raises ValueError: If only one of `text_a_val` and `text_b_val` is given, or if the halves of the pairs have
            different lengths.
        """
        if (text_a_val is None) != (text_b_val is None):
            raise ValueError("Both text_a_val and text_b_val must be provided together, or neither.")
        self._check_aligned(text_a=text_a, text_b=text_b)
        self._check_texts(text_a=text_a, text_b=text_b, text_a_val=text_a_val, text_b_val=text_b_val)
        rows = ColumnRows(text_a=text_a, text_b=text_b)
        if text_a_val is not None and text_b_val is not None:
            self._check_aligned(text_a_val=text_a_val, text_b_val=text_b_val)
            return self._pair_dataset(rows), self._pair_dataset(ColumnRows(text_a=text_a_val, text_b=text_b_val))

        train_indices, val_indices = split_indices(len(rows), test_size, max_test_size=MAX_VALIDATION_SIZE)
        return self._pair_dataset(rows, train_indices), self._pair_dataset(rows, val_indices)

    def fit(
        self: T,
        text_a: Sequence[str],
        text_b: Sequence[str],
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float | int = 0.1,
        device: str = "auto",
        text_a_val: Sequence[str] | None = None,
        text_b_val: Sequence[str] | None = None,
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

        The pairs are read and tokenized per batch. The halves can be lists, or columns of a Hugging Face dataset,
        such as `dataset["query"]`, which are not loaded into memory. The dataset must not have a transform.

        :param text_a: The first half of each training pair.
        :param text_b: The second half of each training pair.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size. If None, a good batch size is chosen automatically.
        :param min_epochs: The minimum number of epochs to train for.
        :param max_epochs: The maximum number of epochs to train for.
            If this is -1, the model trains until early stopping is triggered.
        :param early_stopping_patience: The patience for early stopping.
            If this is None, early stopping is disabled.
        :param test_size: The size of the validation split if `text_a_val` is None: a fraction of the pairs, capped
            at 10,000 pairs, or a number of pairs if it is an int.
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
        self._check_inputs(text_a=text_a, text_b=text_b, text_a_val=text_a_val, text_b_val=text_b_val)
        loss_function = PairInfoNCELoss(temperature=temperature)

        train_dataset, val_dataset = self._create_pair_datasets(text_a, text_b, text_a_val, text_b_val, test_size)
        self._check_pair_splits(len(train_dataset), len(val_dataset))
        batch_size = self._determine_batch_size(batch_size, len(train_dataset))
        if batch_size < 2:
            raise ValueError(f"batch_size must be at least 2, got {batch_size}.")

        self._initialize()
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
