from __future__ import annotations

import copy
import logging
from collections.abc import Sequence, Sized
from typing import Any, TypeVar

import numpy as np
import torch
from datasets import Column, DatasetDict, IterableColumn, IterableDataset, IterableDatasetDict
from datasets import Dataset as HFDataset
from tokenizers import Encoding, Tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from model2vec.inference import StaticModelPipeline
from model2vec.model import DEFAULT_MAX_LENGTH, PathLike, StaticModel, _disable_padding, _get_unk_token_id
from model2vec.train.dataset import ColumnRows, PairDataset, TextDataset, has_only_strings, has_transform
from model2vec.train.trainer import MetricsFn, default_metrics, resolve_device, run_training_loop
from model2vec.train.utils import (
    MAX_VALIDATION_SIZE,
    get_probable_pad_token_id,
    split_indices,
    to_pipeline,
)

logger = logging.getLogger(__name__)


class BaseFinetuneable(nn.Module):
    val_metric = "val_loss"
    early_stopping_direction = "min"

    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        hidden_dim: int = 256,
        n_layers: int = 0,
        out_dim: int = 2,
        pad_id: int = 0,
        token_mapping: list[int] | None = None,
        weights: torch.Tensor | None = None,
        freeze: bool = False,
        normalize: bool = True,
        freeze_weights: bool = False,
        max_length: int | None = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Initialize a trainable StaticModel from a StaticModel.

        :param vectors: The embeddings of the staticmodel.
        :param tokenizer: The tokenizer.
        :param hidden_dim: The hidden dimension of the head.
        :param n_layers: The number of layers in the head. If this is 0 and `out_dim` equals the embedding
            dimension, the model has no head and the embeddings are used as is.
        :param out_dim: The output dimension of the head.
        :param pad_id: The padding id. This is set to 0 in almost all model2vec models
        :param token_mapping: The token mapping. If None, the token mapping is set to the range of the number of vectors.
        :param weights: The weights of the model. If None, the weights are initialized to zeros.
        :param freeze: Whether to freeze the embeddings. This should be set to False in most cases.
        :param normalize: Whether to normalize the embeddings.
        :param freeze_weights: Whether to freeze the learned token weights.
        :param max_length: The default maximum sequence length (in tokens) used to tokenize inputs.
            Matches `StaticModel.max_length`, defaulting to 512.
        """
        super().__init__()
        self.pad_id = pad_id
        self.out_dim = out_dim
        self.embed_dim = vectors.shape[1]
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.normalize = normalize
        self.freeze_weights = freeze_weights
        self.max_length = max_length
        self.token_dropout = 0.0

        self.vectors = vectors
        if self.vectors.dtype != torch.float32:
            dtype = str(self.vectors.dtype)
            logger.warning(
                f"Your vectors are {dtype} precision, converting to to torch.float32 to avoid compatibility issues."
            )
            self.vectors = vectors.float()

        if token_mapping is not None:
            self.token_mapping = torch.tensor(token_mapping, dtype=torch.int64)
        else:
            self.token_mapping = torch.arange(len(vectors), dtype=torch.int64)
        self.token_mapping = nn.Parameter(self.token_mapping, requires_grad=False)
        self.freeze = freeze
        self.embeddings = nn.Embedding.from_pretrained(vectors.clone(), freeze=self.freeze, padding_idx=pad_id)
        self.head = self.construct_head()
        self._weights = weights
        self.w = self.construct_weights()
        # Truncation happens here through `max_length`; a StaticModel's tokenizer carries its own setting.
        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.no_truncation()
        _disable_padding(self.tokenizer)
        self.unk_token_id = _get_unk_token_id(self.tokenizer)

    def _tokenize_ids(self, texts: Sequence[str]) -> list[list[int]]:
        """Tokenize texts into lists of token ids, dropping unknown tokens and truncating to `max_length` tokens.

        :param texts: The texts to tokenize.
        :return: The token ids of each text.
        """
        if self.max_length is not None:
            truncate_length = self.max_length * 10
            texts = [text[:truncate_length] for text in texts]
        encoded: list[Encoding] = self.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        ids = [encoding.ids for encoding in encoded]
        if self.unk_token_id is not None:
            ids = [[token_id for token_id in token_ids if token_id != self.unk_token_id] for token_ids in ids]
        return [token_ids[: self.max_length] for token_ids in ids]

    def _to_targets(self, labels: Any) -> torch.Tensor:
        """Turn a batch of labels, such as vectors, into a float tensor of targets."""
        return torch.as_tensor(labels, dtype=torch.float32)

    def construct_weights(self) -> nn.Parameter:
        """Construct the weights for the model."""
        if self._weights is not None:
            w = self._weights
        else:
            w = torch.ones(len(self.token_mapping)).float()
            w[self.pad_id] = 0
        return nn.Parameter(w, requires_grad=not self.freeze_weights)

    def construct_head(self) -> nn.Sequential:
        """Constructs a simple head, which is empty if it has no layers and doesn't change the dimension."""
        if self.n_layers == 0 and self.embed_dim == self.out_dim:
            return nn.Sequential()
        modules: list[nn.Module] = []
        if self.n_layers == 0:
            modules.append(nn.Linear(self.embed_dim, self.out_dim))
        else:
            # If we have a hidden layer, we should first project to hidden_dim
            modules = [
                nn.Linear(self.embed_dim, self.hidden_dim),
                nn.ReLU(),
            ]
            for _ in range(self.n_layers - 1):
                modules.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
            # We always have a layer mapping from hidden to out.
            modules.append(nn.Linear(self.hidden_dim, self.out_dim))

        linear_modules = [module for module in modules if isinstance(module, nn.Linear)]
        if linear_modules:
            *initial, last = linear_modules
            for module in initial:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            # Final layer does not kaiming
            nn.init.xavier_uniform_(last.weight)
            nn.init.zeros_(last.bias)

        return nn.Sequential(*modules)

    def _initialize(self) -> None:
        """Initialize the classifier for training."""
        self.head = self.construct_head()
        self.embeddings = nn.Embedding.from_pretrained(
            self.vectors.clone(), freeze=self.freeze, padding_idx=self.pad_id
        )
        self.w = self.construct_weights()
        self.train()

    @classmethod
    def from_pretrained(
        cls: type[T],
        path: PathLike = "minishlab/potion-base-32m",
        *,
        token: str | None = None,
        model_name: PathLike | None = None,
        pad_token: str | None = None,
        max_length: int | None = None,
        n_layers: int = 0,
        hidden_dim: int = 256,
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
        :param n_layers: The number of layers in the head.
        :param hidden_dim: The hidden dimension of the head.
        :param out_dim: The output dimension of the head.
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
        n_layers: int = 0,
        hidden_dim: int = 256,
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
        :param n_layers: The number of layers in the head.
        :param hidden_dim: The hidden dimension of the head.
        :param out_dim: The output dimension of the head.
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

    def _apply_token_dropout(self, keep_mask: torch.Tensor) -> torch.Tensor:
        """Randomly zero out a fraction of the kept tokens, leaving at least one token per sample.

        :param keep_mask: A 2D float tensor (batch, seq_len), 1 for real tokens and 0 for padding.
        :return: `keep_mask` with a random subset of real tokens additionally zeroed out.
        """
        if not self.training or self.token_dropout <= 0:
            return keep_mask
        survives = torch.rand_like(keep_mask) >= self.token_dropout
        dropped_mask = keep_mask * survives
        needs_rescue = (dropped_mask.sum(dim=1) == 0) & (keep_mask.sum(dim=1) > 0)
        if needs_rescue.any():
            rescue_idx = keep_mask.argmax(dim=1)
            dropped_mask[needs_rescue, rescue_idx[needs_rescue]] = 1.0
        return dropped_mask

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """A forward pass and mean pooling.

        This function is analogous to `StaticModel.encode`, but reimplemented to allow gradients
        to pass through.

        :param input_ids: A 2D tensor of input ids. All input ids are have to be within bounds.
        :return: The mean over the input ids, weighted by token weights.
        """
        zeros = (input_ids != self.pad_id).float()
        zeros = self._apply_token_dropout(zeros)
        length = zeros.sum(1).clamp(min=1)
        input_ids_embeddings = self.token_mapping[input_ids]
        embedded = self.embeddings(input_ids_embeddings)

        w = self.w[input_ids]
        w = w * zeros
        # Weigh each token
        embedded = torch.bmm(w[:, None, :], embedded).squeeze(1)
        # Mean pooling by dividing by the length
        embedded = embedded / length[:, None]

        if self.normalize:
            return nn.functional.normalize(embedded)
        return embedded

    @torch.no_grad()
    def _encode_single_batch(self, X: list[str]) -> torch.Tensor:
        input_ids = self.tokenize(X)
        return self.head(self._encode(input_ids))

    def encode(self, X: list[str], batch_size: int = 1024, show_progress_bar: bool = False) -> np.ndarray:
        """Encode a single batch of input ids."""
        pred = []
        for batch in trange(0, len(X), batch_size, disable=not show_progress_bar):
            logits = self._encode_single_batch(X[batch : batch + batch_size])
            pred.append(logits.cpu().numpy())

        return np.concatenate(pred, axis=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mean, and a classifier layer after."""
        return self.head(self._encode(input_ids))

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """Tokenize a bunch of strings into a single padded 2D tensor.

        Note that this is not used during training.

        :param texts: The texts to tokenize.
        :return: A 2D padded tensor
        """
        encoded_ids: list[torch.Tensor] = [torch.LongTensor(token_ids) for token_ids in self._tokenize_ids(texts)]
        return pad_sequence(encoded_ids, batch_first=True, padding_value=self.pad_id)

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.embeddings.weight.device

    def to_static_model(self) -> StaticModel:
        """Convert the model to a static model."""
        with torch.no_grad():
            emb = self.embeddings.weight
            emb = emb.cpu().numpy()
            w = self.w.cpu().numpy()

        # If the weights and emb are the same length, the model was not quantized before training.
        if len(w) == len(emb):
            emb = emb * w[:, None]
            return StaticModel(
                vectors=emb,
                weights=None,
                tokenizer=self.tokenizer,
                normalize=self.normalize,
                token_mapping=None,
                max_length=self.max_length,
            )
        return StaticModel(
            vectors=emb,
            weights=w,
            tokenizer=self.tokenizer,
            normalize=self.normalize,
            token_mapping=self.token_mapping.numpy(),
            max_length=self.max_length,
        )

    def to_pipeline(self) -> StaticModelPipeline:
        """Convert the model to an inference pipeline."""
        return to_pipeline(self)

    def _determine_batch_size(self, batch_size: int | None, train_length: int) -> int:
        if batch_size is None:
            # Set to a multiple of 32
            base_number = int(min(max(1, (train_length / 30) // 32), 16))
            batch_size = int(base_number * 32)
            logger.info("Batch size automatically set to %d.", batch_size)

        return batch_size

    def _train(
        self,
        loss_function: nn.Module,
        learning_rate: float,
        train_dataset: TextDataset | PairDataset,
        val_dataset: TextDataset | PairDataset,
        batch_size: int,
        early_stopping_patience: int | None,
        min_epochs: int | None,
        max_epochs: int | None,
        device: str,
        validation_steps: int | None,
        compute_metrics: MetricsFn = default_metrics,
        token_dropout: float = 0.0,
    ) -> None:
        if not 0.0 <= token_dropout < 1.0:
            raise ValueError("token_dropout must be in the range [0, 1).")
        self.token_dropout = token_dropout

        val_check_interval, check_val_every_epoch = self._determine_val_check_interval(
            validation_steps, len(train_dataset), batch_size
        )

        state_dict = run_training_loop(
            model=self,
            loss_function=loss_function,
            learning_rate=learning_rate,
            val_metric=self.val_metric,
            early_stopping_direction=self.early_stopping_direction,
            train_loader=train_dataset.to_dataloader(shuffle=True, batch_size=batch_size),
            val_loader=val_dataset.to_dataloader(shuffle=False, batch_size=batch_size),
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            device=resolve_device(device),
            val_check_interval=val_check_interval,
            check_val_every_epoch=check_val_every_epoch,
            compute_metrics=compute_metrics,
        )

        self.load_state_dict(state_dict)
        self.to("cpu")
        self.eval()

    @staticmethod
    def _determine_val_check_interval(
        validation_steps: int | None, train_length: int, batch_size: int
    ) -> tuple[int | None, int | None]:
        val_check_interval: int | None = None
        check_val_every_epoch: int | None = 1
        if validation_steps is None:
            n_train_batches = train_length // batch_size
            target_checks_per_epoch = 4
            min_train_steps_between_val = 250

            # If we have more than 250 batches, smoothly interpolate
            if n_train_batches > min_train_steps_between_val:
                val_check_interval = max(
                    min_train_steps_between_val,
                    n_train_batches // target_checks_per_epoch,
                )
                check_val_every_epoch = None
        else:
            val_check_interval = validation_steps
            check_val_every_epoch = None

        return val_check_interval, check_val_every_epoch

    @staticmethod
    def _check_inputs(**arguments: object) -> None:
        """Check that every argument of `fit` is a sequence, an array, a tensor, or a column of a Hugging Face dataset.

        :param **arguments: The arguments, by name. None is skipped.
        :raises ValueError: If an argument is a Hugging Face `Dataset` or `DatasetDict`, an iterable dataset or one of
            its columns, a column of a dataset with a transform, a single string, or any other object that is not a
            sequence, an array, or a tensor.
        """
        for name, value in arguments.items():
            if isinstance(value, (IterableDataset, IterableDatasetDict, IterableColumn)):
                raise ValueError(
                    f"{name} comes from an iterable Hugging Face dataset, which has no length. Pass a column of a "
                    "regular dataset instead, such as one loaded without `streaming=True`."
                )
            if isinstance(value, (HFDataset, DatasetDict)):
                raise ValueError(
                    f"{name} is a Hugging Face dataset. Pass one of its columns instead, such as `dataset['text']`."
                )
            if isinstance(value, Column) and has_transform(value):
                raise ValueError(
                    f"{name} is a column of a Hugging Face dataset with a transform. Apply the transform first with "
                    "`dataset.map(transform, batched=True)`, or pass a list."
                )
            if isinstance(value, str) or (
                value is not None and not isinstance(value, (Sequence, np.ndarray, torch.Tensor))
            ):
                raise ValueError(
                    f"{name} must be a list, a tuple, an array, a tensor, or a column of a Hugging Face dataset, got "
                    f"{type(value).__name__}."
                )

    @staticmethod
    def _check_aligned(**columns: Sized) -> None:
        """Check that the columns of the training or validation data have the same length.

        :param **columns: The columns, by name.
        :raises ValueError: If the columns don't all have the same length.
        """
        lengths = {name: len(column) for name, column in columns.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"{' and '.join(lengths)} must have the same length, got {lengths}.")

    @staticmethod
    def _check_texts(**texts: Sequence[str] | None) -> None:
        """Check that all texts are strings.

        :param **texts: The texts, by name. None is skipped.
        :raises ValueError: If a text is missing or is not a string.
        """
        for name, values in texts.items():
            if values is None:
                continue
            is_valid = (
                has_only_strings(values)
                if isinstance(values, Column)
                else all(isinstance(text, str) for text in values)
            )
            if not is_valid:
                raise ValueError(f"All texts in {name} must be strings.")

    def _text_dataset(self, rows: ColumnRows, indices: np.ndarray | None = None) -> TextDataset:
        """Create a dataset of labeled texts that are tokenized per batch.

        :param rows: The labeled texts, in a `text` and a `label` column.
        :param indices: The indices of the rows that belong to the dataset. If None, all rows belong to it.
        :return: The dataset.
        """
        return TextDataset(rows, self._tokenize_ids, self._to_targets, indices, pad_id=self.pad_id)

    def _create_datasets(
        self,
        X: Sequence[str],
        y: Any,
        X_val: Sequence[str] | None,
        y_val: Any | None,
        test_size: float | int,
        stratify_by: Sequence[Any] | None = None,
    ) -> tuple[TextDataset, TextDataset]:
        """Create the training and validation datasets.

        :param X: The training texts.
        :param y: The training labels.
        :param X_val: The validation texts. If None, the validation data is split off from `X` and `y`.
        :param y_val: The validation labels.
        :param test_size: The size of the validation split if `X_val` is None: a fraction of the data, capped at
            `MAX_VALIDATION_SIZE` rows, or a number of rows if it is an int.
        :param stratify_by: Validated single labels to stratify the validation split by. If None, the split is not
            stratified.
        :return: The train and validation datasets.
        :raises ValueError: If only one of `X_val` and `y_val` is given, or if the texts and labels have different
            lengths.
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("Both X_val and y_val must be provided together, or neither.")
        self._check_aligned(X=X, y=y)
        self._check_texts(X=X, X_val=X_val)
        rows = ColumnRows(text=X, label=y)
        if X_val is not None and y_val is not None:
            self._check_aligned(X_val=X_val, y_val=y_val)
            return self._text_dataset(rows), self._text_dataset(ColumnRows(text=X_val, label=y_val))

        train_indices, val_indices = split_indices(
            len(rows), test_size, max_test_size=MAX_VALIDATION_SIZE, stratify_by=stratify_by
        )
        return self._text_dataset(rows, train_indices), self._text_dataset(rows, val_indices)


T = TypeVar("T", bound=BaseFinetuneable)


def _load_static_model(path: PathLike, *, token: str | None, model_name: PathLike | None) -> StaticModel:
    """Load a static model, resolving the deprecated `model_name` argument.

    :param path: The path to the folder containing the model, or a repository on the Hugging Face Hub.
    :param token: The token to use to download the model from the hub.
    :param model_name: Deprecated alias for `path`. If given, it overrides `path`.
    :return: The loaded static model.
    """
    if model_name is not None:
        logger.warning("The 'model_name' argument is deprecated. Use 'path' instead.")
        path = model_name
    return StaticModel.from_pretrained(path, token=token)


def _static_model_arguments(model: StaticModel, *, pad_token: str | None, max_length: int | None) -> dict[str, Any]:
    """Derive the constructor arguments of a finetuneable model from a static model.

    :param model: The static model to derive the arguments from.
    :param pad_token: The token to use for padding. If None, it is inferred from the tokenizer.
    :param max_length: The default maximum sequence length to use for tokenization. If None, the
        static model's `max_length` is used.
    :return: The constructor arguments.
    """
    model.embedding = np.nan_to_num(model.embedding)
    weights = torch.from_numpy(model.weights) if model.weights is not None else None
    token_mapping = model.token_mapping.tolist() if model.token_mapping is not None else None
    if pad_token is not None:
        pad_id = model.tokenizer.get_vocab()[pad_token]
    else:
        pad_id = get_probable_pad_token_id(model.tokenizer)
    return {
        "vectors": torch.from_numpy(model.embedding),
        "pad_id": pad_id,
        "tokenizer": model.tokenizer,
        "token_mapping": token_mapping,
        "weights": weights,
        "max_length": model.max_length if max_length is None else max_length,
    }
