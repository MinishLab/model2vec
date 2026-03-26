from __future__ import annotations

import logging
from itertools import chain
from typing import Any, cast

import lightning as pl
import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import trange

from model2vec.inference import evaluate_single_or_multi_label
from model2vec.train.base import _BaseFinetuneable
from model2vec.train.lightning_modules import ClassifierLightningModule, MultiLabelClassifierLightningModule

logger = logging.getLogger(__name__)

_DEFAULT_RANDOM_SEED = 42
LabelType = list[str] | list[list[str]]


class StaticModelForClassification(_BaseFinetuneable):
    val_metric = "val_accuracy"
    early_stopping_direction = "max"

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
    ) -> None:
        """Initialize a standard classifier model."""
        # Alias: Follows scikit-learn. Set to dummy classes
        self.classes_: list[str] = [str(x) for x in range(out_dim)]
        # multilabel flag will be set based on the type of `y` passed to fit.
        self.multilabel: bool = False
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
        )

    @property
    def classes(self) -> np.ndarray:
        """Return all clasess in the correct order."""
        return np.array(self.classes_)

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
            logits = self._encode_single_batch(X[batch : batch + batch_size])
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

    def predict_proba(self, X: list[str], show_progress_bar: bool = False, batch_size: int = 1024) -> np.ndarray:
        """
        Predict probabilities for each class.

        In single-label mode, returns softmax probabilities.
        In multilabel mode, returns sigmoid probabilities.
        """
        pred = []
        for batch in trange(0, len(X), batch_size, disable=not show_progress_bar):
            logits = self._encode_single_batch(X[batch : batch + batch_size])
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
        class_weight: torch.Tensor | None = None,
        random_seed: int = _DEFAULT_RANDOM_SEED,
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
        :param class_weight: The weight of the classes. If None, all classes are weighted equally. Must
            have the same length as the number of classes.
        :param random_seed: The random seed to use. Defaults to 42.
        :return: The fitted model.
        :raises ValueError: If either X_val or y_val are provided, but not both.
        """
        pl.seed_everything(random_seed)
        logger.info("Re-initializing model.")

        # Determine whether the task is multilabel based on the type of y.
        self._initialize_on_labels(y)
        self._initialize()

        if class_weight is not None:
            if len(class_weight) != len(self.classes_):
                raise ValueError("class_weight must have the same length as the number of classes.")

        train_dataset, val_dataset = self._create_datasets(X, y, X_val, y_val, test_size)
        batch_size = self._determine_batch_size(batch_size, len(train_dataset))

        c: pl.LightningModule
        if self.multilabel:
            c = MultiLabelClassifierLightningModule(self, learning_rate=learning_rate, class_weight=class_weight)
        else:
            c = ClassifierLightningModule(self, learning_rate=learning_rate, class_weight=class_weight)

        self._train(
            module=c,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            device=device,
        )

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

    def _initialize_on_labels(self, y: LabelType) -> None:
        """
        Sets the output dimensionality, the classes, and initializes the head.

        :param y: The labels.
        :raises ValueError: If the labels are inconsistent.
        """
        if isinstance(y[0], (str, int)):
            y = cast(list[str], y)
            # Check if all labels are strings or integers.
            if not all(isinstance(label, (str, int)) for label in y):
                raise ValueError("Inconsistent label types in y. All labels must be strings or integers.")
            self.multilabel = False
            classes = sorted(set(y))
        else:
            y = cast(list[list[str]], y)
            # Check if all labels are lists or tuples.
            if not all(isinstance(label, (list, tuple)) for label in y):
                raise ValueError("Inconsistent label types in y. All labels must be lists or tuples.")
            self.multilabel = True
            classes = sorted(set(chain.from_iterable(y)))

        self.classes_ = classes
        self.out_dim = len(self.classes_)

    def _labels_to_tensor(self, labels: Any) -> torch.Tensor:
        """Convert a list or list of list of labels to a tensor."""
        if self.multilabel:
            # Convert labels to multi-hot vectors
            num_classes = len(self.classes_)
            labels_tensor = torch.zeros(len(labels), num_classes, dtype=torch.float)
            mapping = {label: idx for idx, label in enumerate(self.classes_)}
            for i, sample_labels in enumerate(labels):
                indices = [mapping[label] for label in sample_labels]
                labels_tensor[i, indices] = 1.0
        else:
            labels_tensor = torch.tensor(
                [self.classes_.index(label) for label in cast(list[str], labels)], dtype=torch.long
            )

        return labels_tensor
