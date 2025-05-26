from __future__ import annotations

import logging
from collections import Counter
from itertools import chain
from tempfile import TemporaryDirectory
from typing import TypeVar, cast

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from tokenizers import Tokenizer
from torch import nn
from tqdm import trange

from model2vec.inference import StaticModelPipeline, evaluate_single_or_multi_label
from model2vec.train.base import FinetunableStaticModel, TextDataset

logger = logging.getLogger(__name__)
_RANDOM_SEED = 42

LabelType = TypeVar("LabelType", list[str], list[list[str]])


class StaticModelForClassification(FinetunableStaticModel):
    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        n_layers: int = 1,
        hidden_dim: int = 512,
        out_dim: int = 2,
        pad_id: int = 0,
    ) -> None:
        """Initialize a standard classifier model."""
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # Alias: Follows scikit-learn. Set to dummy classes
        self.classes_: list[str] = [str(x) for x in range(out_dim)]
        # multilabel flag will be set based on the type of `y` passed to fit.
        self.multilabel: bool = False
        super().__init__(vectors=vectors, out_dim=out_dim, pad_id=pad_id, tokenizer=tokenizer)

    @property
    def classes(self) -> np.ndarray:
        """Return all clasess in the correct order."""
        return np.array(self.classes_)

    def construct_head(self) -> nn.Sequential:
        """Constructs a simple classifier head."""
        if self.n_layers == 0:
            return nn.Sequential(nn.Linear(self.embed_dim, self.out_dim))
        modules = [
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(self.n_layers - 1):
            modules.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        modules.extend([nn.Linear(self.hidden_dim, self.out_dim)])

        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        return nn.Sequential(*modules)

    def predict(
        self, X: list[str], show_progress_bar: bool = False, batch_size: int = 1024, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict labels for a set of texts.

        In single-label mode, each prediction is a single class.
        In multilabel mode, each prediction is a list of classes.

        :param X: The texts to predict on.
        :param show_progress_bar: Whether to show a progress bar.
        :param batch_size: The batch size.
        :param threshold: The threshold for multilabel classification.
        :return: The predictions.
        """
        pred = []
        for batch in trange(0, len(X), batch_size, disable=not show_progress_bar):
            logits = self._predict_single_batch(X[batch : batch + batch_size])
            if self.multilabel:
                probs = torch.sigmoid(logits)
                mask = (probs > threshold).cpu().numpy()
                pred.extend([self.classes[np.flatnonzero(row)] for row in mask])
            else:
                pred.extend([self.classes[idx] for idx in logits.argmax(dim=1).tolist()])
        if self.multilabel:
            # Return as object array to allow for lists of varying lengths.
            return np.array(pred, dtype=object)
        else:
            return np.array(pred)

    @torch.no_grad()
    def _predict_single_batch(self, X: list[str]) -> torch.Tensor:
        input_ids = self.tokenize(X)
        vectors, _ = self.forward(input_ids)
        return vectors

    def predict_proba(self, X: list[str], show_progress_bar: bool = False, batch_size: int = 1024) -> np.ndarray:
        """
        Predict probabilities for each class.

        In single-label mode, returns softmax probabilities.
        In multilabel mode, returns sigmoid probabilities.
        """
        pred = []
        for batch in trange(0, len(X), batch_size, disable=not show_progress_bar):
            logits = self._predict_single_batch(X[batch : batch + batch_size])
            if self.multilabel:
                pred.append(torch.sigmoid(logits).cpu().numpy())
            else:
                pred.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(pred, axis=0)

    def fit(
        self,
        X: list[str],
        y: LabelType,
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
        X_val: list[str] | None = None,
        y_val: LabelType | None = None,
    ) -> StaticModelForClassification:
        """
        Fit a model.

        This function creates a Lightning Trainer object and fits the model to the data.
        It supports both single-label and multi-label classification.
        We use early stopping. After training, the weights of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        If `X_val` and `y_val` are not provided, the function will automatically
        split the training data into a train and validation set using `test_size`.

        :param X: The texts to train on.
        :param y: The labels to train on. If the first element is a list, multi-label classification is assumed.
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
        :param y_val: The labels to be used for validation.
        :return: The fitted model.
        :raises ValueError: If either X_val or y_val are provided, but not both.
        """
        pl.seed_everything(_RANDOM_SEED)
        logger.info("Re-initializing model.")

        # Determine whether the task is multilabel based on the type of y.

        self._initialize(y)

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
            train_texts, validation_texts, train_labels, validation_labels = self._train_test_split(
                X,
                y,
                test_size=test_size,
            )

        if batch_size is None:
            # Set to a multiple of 32
            base_number = int(min(max(1, (len(train_texts) / 30) // 32), 16))
            batch_size = int(base_number * 32)
            logger.info("Batch size automatically set to %d.", batch_size)

        logger.info("Preparing train dataset.")
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        logger.info("Preparing validation dataset.")
        val_dataset = self._prepare_dataset(validation_texts, validation_labels)

        c = _ClassifierLightningModule(self, learning_rate=learning_rate)

        n_train_batches = len(train_dataset) // batch_size
        callbacks: list[Callback] = []
        if early_stopping_patience is not None:
            callback = EarlyStopping(monitor="val_accuracy", mode="max", patience=early_stopping_patience)
            callbacks.append(callback)

        # If the dataset is small, we check the validation set every epoch.
        # If the dataset is large, we check the validation set every 250 batches.
        if n_train_batches < 250:
            val_check_interval = None
            check_val_every_epoch = 1
        else:
            val_check_interval = max(250, 2 * len(val_dataset) // batch_size)
            check_val_every_epoch = None

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
                c,
                train_dataloaders=train_dataset.to_dataloader(shuffle=True, batch_size=batch_size),
                val_dataloaders=val_dataset.to_dataloader(shuffle=False, batch_size=batch_size),
            )
            best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            best_model_weights = torch.load(best_model_path, weights_only=True)

        state_dict = {}
        for weight_name, weight in best_model_weights["state_dict"].items():
            state_dict[weight_name.removeprefix("model.")] = weight

        self.load_state_dict(state_dict)
        self.eval()
        return self

    def evaluate(
        self, X: list[str], y: LabelType, batch_size: int = 1024, threshold: float = 0.5, output_dict: bool = False
    ) -> str | dict[str, dict[str, float]]:
        """
        Evaluate the classifier on a given dataset using scikit-learn's classification report.

        :param X: The texts to predict on.
        :param y: The ground truth labels.
        :param batch_size: The batch size.
        :param threshold: The threshold for multilabel classification.
        :param output_dict: Whether to output the classification report as a dictionary.
        :return: A classification report.
        """
        self.eval()
        predictions = self.predict(X, show_progress_bar=True, batch_size=batch_size, threshold=threshold)
        report = evaluate_single_or_multi_label(predictions=predictions, y=y, output_dict=output_dict)

        return report

    def _initialize(self, y: LabelType) -> None:
        """
        Sets the output dimensionality, the classes, and initializes the head.

        :param y: The labels.
        :raises ValueError: If the labels are inconsistent.
        """
        if isinstance(y[0], (str, int)):
            # Check if all labels are strings or integers.
            if not all(isinstance(label, (str, int)) for label in y):
                raise ValueError("Inconsistent label types in y. All labels must be strings or integers.")
            self.multilabel = False
            classes = sorted(set(y))
        else:
            # Check if all labels are lists or tuples.
            if not all(isinstance(label, (list, tuple)) for label in y):
                raise ValueError("Inconsistent label types in y. All labels must be lists or tuples.")
            self.multilabel = True
            classes = sorted(set(chain.from_iterable(y)))

        self.classes_ = classes
        self.out_dim = len(self.classes_)  # Update output dimension
        self.head = self.construct_head()
        self.embeddings = nn.Embedding.from_pretrained(self.vectors.clone(), freeze=False, padding_idx=self.pad_id)
        self.w = self.construct_weights()
        self.train()

    def _prepare_dataset(self, X: list[str], y: LabelType, max_length: int = 512) -> TextDataset:
        """
        Prepare a dataset. For multilabel classification, each target is converted into a multi-hot vector.

        :param X: The texts.
        :param y: The labels.
        :param max_length: The maximum length of the input.
        :return: A TextDataset.
        """
        # This is a speed optimization.
        # assumes a mean token length of 10, which is really high, so safe.
        truncate_length = max_length * 10
        X = [x[:truncate_length] for x in X]
        tokenized: list[list[int]] = [
            encoding.ids[:max_length] for encoding in self.tokenizer.encode_batch_fast(X, add_special_tokens=False)
        ]
        if self.multilabel:
            # Convert labels to multi-hot vectors
            num_classes = len(self.classes_)
            labels_tensor = torch.zeros(len(y), num_classes, dtype=torch.float)
            mapping = {label: idx for idx, label in enumerate(self.classes_)}
            for i, sample_labels in enumerate(y):
                indices = [mapping[label] for label in sample_labels]
                labels_tensor[i, indices] = 1.0
        else:
            labels_tensor = torch.tensor([self.classes_.index(label) for label in cast(list[str], y)], dtype=torch.long)
        return TextDataset(tokenized, labels_tensor)

    def _train_test_split(
        self,
        X: list[str],
        y: list[str] | list[list[str]],
        test_size: float,
    ) -> tuple[list[str], list[str], LabelType, LabelType]:
        """
        Split the data.

        For single-label classification, stratification is attempted (if possible).
        For multilabel classification, a random split is performed.
        """
        if not self.multilabel:
            label_counts = Counter(y)
            if min(label_counts.values()) < 2:
                logger.info("Some classes have less than 2 samples. Stratification is disabled.")
                return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
            return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y)
        else:
            # Multilabel classification does not support stratification.
            return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

    def to_pipeline(self) -> StaticModelPipeline:
        """Convert the model to an sklearn pipeline."""
        static_model = self.to_static_model()

        random_state = np.random.RandomState(_RANDOM_SEED)
        n_items = len(self.classes)
        X = random_state.randn(n_items, static_model.dim)
        y = self.classes

        converted = make_pipeline(MLPClassifier(hidden_layer_sizes=(self.hidden_dim,) * self.n_layers))
        converted.fit(X, y)
        mlp_head: MLPClassifier = converted[-1]

        for index, layer in enumerate([module for module in self.head if isinstance(module, nn.Linear)]):
            mlp_head.coefs_[index] = layer.weight.detach().cpu().numpy().T
            mlp_head.intercepts_[index] = layer.bias.detach().cpu().numpy()
        # Below is necessary to ensure that the converted model works correctly.
        # In scikit-learn, a binary classifier only has a single vector of output coefficients
        # and a single intercept. We use two output vectors.
        # To convert correctly, we need to set the outputs correctly, and fix the activation function.
        # Make sure n_outputs is set to > 1.
        mlp_head.n_outputs_ = self.out_dim
        # Set to softmax or sigmoid
        mlp_head.out_activation_ = "logistic" if self.multilabel else "softmax"

        return StaticModelPipeline(static_model, converted)


class _ClassifierLightningModule(pl.LightningModule):
    def __init__(self, model: StaticModelForClassification, learning_rate: float) -> None:
        """Initialize the LightningModule."""
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss() if not model.multilabel else nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step using cross-entropy loss for single-label and binary cross-entropy for multilabel training."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step computing loss and accuracy."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        if self.model.multilabel:
            preds = (torch.sigmoid(head_out) > 0.5).float()
            # Multilabel accuracy is defined as the Jaccard score averaged over samples.
            accuracy = jaccard_score(y.cpu(), preds.cpu(), average="samples")
        else:
            accuracy = (head_out.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            threshold=0.03,
            threshold_mode="rel",
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
