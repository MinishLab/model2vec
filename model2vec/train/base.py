from __future__ import annotations

import logging
from collections.abc import Sequence
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback, EarlyStopping
from tokenizers import Encoding, Tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from model2vec import StaticModel
from model2vec.inference import StaticModelPipeline
from model2vec.train.dataset import TextDataset
from model2vec.train.utils import (
    get_probable_pad_token_id,
    logit,
    suppress_lightning_warnings,
    to_pipeline,
    train_test_split,
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
    ) -> None:
        """
        Initialize a trainable StaticModel from a StaticModel.

        :param vectors: The embeddings of the staticmodel.
        :param tokenizer: The tokenizer.
        :param hidden_dim: The hidden dimension of the head.
        :param n_layers: The number of layers in the head.
        :param out_dim: The output dimension of the head.
        :param pad_id: The padding id. This is set to 0 in almost all model2vec models
        :param token_mapping: The token mapping. If None, the token mapping is set to the range of the number of vectors.
        :param weights: The weights of the model. If None, the weights are initialized to zeros.
        :param freeze: Whether to freeze the embeddings. This should be set to False in most cases.
        :param normalize: Whether to normalize the embeddings.
        """
        super().__init__()
        self.pad_id = pad_id
        self.out_dim = out_dim
        self.embed_dim = vectors.shape[1]
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.normalize = normalize

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
        self.tokenizer = tokenizer

    def construct_weights(self) -> nn.Parameter:
        """Construct the weights for the model."""
        if self._weights is not None:
            w = logit(self._weights)
            return nn.Parameter(w.float(), requires_grad=True)
        weights = torch.zeros(len(self.token_mapping))
        weights[self.pad_id] = -10_000
        return nn.Parameter(weights, requires_grad=not self.freeze)

    def construct_head(self) -> nn.Sequential:
        """Constructs a simple classifier head."""
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
        cls: type[ModelType], path: str = "minishlab/potion-base-32m", *, token: str | None = None, **kwargs: Any
    ) -> ModelType:
        """Load the model from a pretrained model2vec model."""
        if model_name := kwargs.pop("model_name", None):
            logger.warning("The 'model_name' argument is deprecated. Use 'path' instead.")
            path = model_name
        model = StaticModel.from_pretrained(path, token=token)
        return cls.from_static_model(model=model, **kwargs)

    @classmethod
    def from_static_model(
        cls: type[ModelType], *, model: StaticModel, pad_token: str | None = None, **kwargs: Any
    ) -> ModelType:
        """Load the model from a static model."""
        model.embedding = np.nan_to_num(model.embedding)
        weights = torch.from_numpy(model.weights) if model.weights is not None else None
        embeddings_converted = torch.from_numpy(model.embedding)
        if model.token_mapping is not None:
            token_mapping = model.token_mapping.tolist()
        else:
            token_mapping = None
        if pad_token is not None:
            pad_id = model.tokenizer.get_vocab()[pad_token]
        else:
            pad_id = get_probable_pad_token_id(model.tokenizer)
        return cls(
            vectors=embeddings_converted,
            pad_id=pad_id,
            tokenizer=model.tokenizer,
            token_mapping=token_mapping,
            weights=weights,
            **kwargs,
        )

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        A forward pass and mean pooling.

        This function is analogous to `StaticModel.encode`, but reimplemented to allow gradients
        to pass through.

        :param input_ids: A 2D tensor of input ids. All input ids are have to be within bounds.
        :return: The mean over the input ids, weighted by token weights.
        """
        w = self.w[input_ids]
        w = torch.sigmoid(w)
        zeros = (input_ids != self.pad_id).float()
        w = w * zeros
        # Add a small epsilon to avoid division by zero
        length = zeros.sum(1) + 1e-16
        input_ids_embeddings = self.token_mapping[input_ids]
        embedded = self.embeddings(input_ids_embeddings)
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

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean, and a classifier layer after."""
        encoded = self._encode(input_ids)
        return self.head(encoded), encoded

    def tokenize(self, texts: list[str], max_length: int | None = 512) -> torch.Tensor:
        """
        Tokenize a bunch of strings into a single padded 2D tensor.

        Note that this is not used during training.

        :param texts: The texts to tokenize.
        :param max_length: If this is None, the sequence lengths are truncated to 512.
        :return: A 2D padded tensor
        """
        encoded: list[Encoding] = self.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        encoded_ids: list[torch.Tensor] = [torch.Tensor(encoding.ids[:max_length]).long() for encoding in encoded]
        return pad_sequence(encoded_ids, batch_first=True, padding_value=self.pad_id)

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.embeddings.weight.device

    def to_static_model(self) -> StaticModel:
        """Convert the model to a static model."""
        emb = self.embeddings.weight.detach().cpu().numpy()
        w = torch.sigmoid(self.w).detach().cpu().numpy()
        # If the weights and emb are the same length, the model was not quantized before training.
        if len(w) == len(emb):
            emb = emb * w[:, None]
            return StaticModel(
                vectors=emb, weights=None, tokenizer=self.tokenizer, normalize=self.normalize, token_mapping=None
            )
        return StaticModel(
            vectors=emb,
            weights=w,
            tokenizer=self.tokenizer,
            normalize=self.normalize,
            token_mapping=self.token_mapping.numpy(),
        )

    def to_pipeline(self) -> StaticModelPipeline:
        """Convert the model to an sklearn pipeline."""
        return to_pipeline(self)

    def _determine_batch_size(self, batch_size: int | None, train_length: int) -> int:
        if batch_size is None:
            # Set to a multiple of 32
            base_number = int(min(max(1, (train_length / 30) // 32), 16))
            batch_size = int(base_number * 32)
            logger.info("Batch size automatically set to %d.", batch_size)

        return batch_size

    def _check_val_split(
        self, X: list[str], y: list, X_val: list[str] | None, y_val: list | None, test_size: float
    ) -> tuple[list[str], list[str], Sequence, Sequence]:
        if (X_val is not None) != (y_val is not None):
            raise ValueError("Both X_val and y_val must be provided together, or neither.")

        if X_val is not None and y_val is not None:
            # Additional check to ensure y_val is of the same type as y
            if type(y_val[0]) != type(y[0]):
                raise ValueError("X_val and y_val must be of the same type as X and y.")

            train_texts = X
            train_labels = y
            validation_texts = X_val
            validation_labels = y_val
        else:
            train_texts, validation_texts, train_labels, validation_labels = train_test_split(X, y, test_size=test_size)

        return train_texts, validation_texts, train_labels, validation_labels

    @suppress_lightning_warnings
    def _train(
        self,
        module: LightningModule,
        train_dataset: TextDataset,
        val_dataset: TextDataset,
        batch_size: int,
        early_stopping_patience: int | None,
        min_epochs: int | None,
        max_epochs: int | None,
        device: str,
        validation_steps: int | None,
    ) -> None:
        callbacks: list[Callback] = []
        if early_stopping_patience is not None:
            callback = EarlyStopping(
                monitor=self.val_metric,
                mode=self.early_stopping_direction,
                patience=early_stopping_patience,
                min_delta=0.001,
            )
            callbacks.append(callback)

        val_check_interval, check_val_every_epoch = self._determine_val_check_interval(
            validation_steps, len(train_dataset), batch_size
        )

        with TemporaryDirectory() as tempdir:
            trainer = pl.Trainer(
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                callbacks=callbacks,
                val_check_interval=val_check_interval,
                check_val_every_n_epoch=check_val_every_epoch,
                accelerator=device,
                default_root_dir=tempdir,
            )

            trainer.fit(
                module,
                train_dataloaders=train_dataset.to_dataloader(shuffle=True, batch_size=batch_size),
                val_dataloaders=val_dataset.to_dataloader(shuffle=False, batch_size=batch_size),
            )
            best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            best_model_weights = torch.load(best_model_path, weights_only=True)

        state_dict = {}
        for weight_name, weight in best_model_weights["state_dict"].items():
            if "loss_function" in weight_name:
                # Skip the loss function class weight as its not needed for predictions
                continue
            state_dict[weight_name.removeprefix("model.")] = weight

        self.load_state_dict(state_dict)
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

    def _prepare_dataset(self, X: list[str], y: torch.Tensor, max_length: int = 512) -> TextDataset:
        """
        Prepare a dataset.

        :param X: The texts.
        :param y: The labels.
        :param max_length: The maximum length of the input.
        :return: A TextDataset.
        """
        # This is a speed optimization.
        # assumes a mean token length of 10, which is really high, so safe.
        truncate_length = max_length * 10
        batch_size = 1024
        tokenized: list[list[int]] = []
        for batch_idx in trange(0, len(X), 1024, desc="Tokenizing data"):
            batch = [x[:truncate_length] for x in X[batch_idx : batch_idx + batch_size]]
            encoded = self.tokenizer.encode_batch_fast(batch, add_special_tokens=False)
            tokenized.extend([encoding.ids[:max_length] for encoding in encoded])

        return TextDataset(tokenized, y)

    def _labels_to_tensor(self, labels: Any) -> torch.Tensor:
        """Turn the labels into a tensor."""
        return labels

    def _create_datasets(
        self,
        X: list[str],
        y: Any,
        X_val: list[str] | None,
        y_val: Any | None,
        test_size: float,
    ) -> tuple[TextDataset, TextDataset]:
        train_texts, validation_texts, train_labels, validation_labels = self._check_val_split(
            X, y, X_val, y_val, test_size
        )
        y_tensor = self._labels_to_tensor(train_labels)
        y_val_tensor = self._labels_to_tensor(validation_labels)

        logger.info("Preparing train dataset.")
        train_dataset = self._prepare_dataset(train_texts, y_tensor)
        logger.info("Preparing validation dataset.")
        val_dataset = self._prepare_dataset(validation_texts, y_val_tensor)

        return train_dataset, val_dataset


ModelType = TypeVar("ModelType", bound=BaseFinetuneable)
