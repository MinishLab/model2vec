from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, Literal, cast

import numpy as np
import torch
from datasets import Column
from tokenizers import Tokenizer
from torch import nn
from tqdm import trange

from model2vec.inference import evaluate_single_or_multi_label
from model2vec.model import DEFAULT_MAX_LENGTH
from model2vec.train.base import BaseFinetuneable
from model2vec.train.dataset import read_label_column
from model2vec.train.utils import DEFAULT_RANDOM_SEED, seed_everything

logger = logging.getLogger(__name__)

LabelType = list[str] | list[list[str]]


def _classifier_metrics(head_out: torch.Tensor, y: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
    """Metrics for single-label classification: loss and accuracy."""
    accuracy = (head_out.argmax(dim=1) == y).float().mean()
    return {"loss": loss.item(), "accuracy": accuracy.item()}


def _compute_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute the multilabel accuracy score, averaged over samples."""
    intersection = (y_true * y_pred).sum(dim=1)
    union = ((y_true + y_pred) > 0).float().sum(dim=1)
    scores = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    return scores.mean().item()


def _multilabel_classifier_metrics(head_out: torch.Tensor, y: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
    """Metrics for multi-label classification: loss and Jaccard accuracy."""
    preds = (torch.sigmoid(head_out) > 0.5).float()
    accuracy = _compute_accuracy(y, preds)
    return {"loss": loss.item(), "accuracy": accuracy}


def _read_labels(y: LabelType, name: str) -> tuple[bool, Counter]:
    """Determine whether labels are multi-label, and count the number of times each class occurs.

    :param y: The labels. If the first label is a list, multi-label classification is assumed. A column of a
        Hugging Face dataset is read in batches, without converting it to Python objects.
    :param name: The name of the labels, used in error messages.
    :return: Whether the labels are multi-label, and the number of times each class occurs.
    :raises ValueError: If the labels are inconsistent, or are not strings, integers, or lists of those.
    """
    if isinstance(y, Column):
        return read_label_column(y, name)
    if isinstance(y, (np.ndarray, torch.Tensor)):
        y = y.tolist()

    if isinstance(y[0], (str, int)):
        if not all(isinstance(label, (str, int)) for label in y):
            raise ValueError(f"Inconsistent label types in {name}. All labels must be strings or integers.")
        return False, Counter(cast(list[str], y))
    if not all(isinstance(label, (list, tuple)) for label in y):
        raise ValueError(f"Inconsistent label types in {name}. All labels must be lists or tuples.")
    classes = list(chain.from_iterable(cast(list[list[str]], y)))
    if not all(isinstance(label, (str, int)) for label in classes):
        raise ValueError(f"Inconsistent label types in {name}. All classes must be strings or integers.")
    return True, Counter(classes)


class StaticModelForClassification(BaseFinetuneable):
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
        normalize: bool = True,
        freeze_weights: bool = False,
        max_length: int | None = DEFAULT_MAX_LENGTH,
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
            normalize=normalize,
            freeze_weights=freeze_weights,
            max_length=max_length,
        )

    @property
    def classes(self) -> np.ndarray:
        """Return all clasess in the correct order."""
        return np.array(self.classes_)

    def construct_head(self) -> nn.Sequential:
        """Constructs a classifier head, which always has at least one linear layer."""
        if self.n_layers == 0:
            linear = nn.Linear(self.embed_dim, self.out_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            return nn.Sequential(linear)
        return super().construct_head()

    def predict(
        self, X: list[str], show_progress_bar: bool = False, batch_size: int = 1024, threshold: float = 0.5
    ) -> np.ndarray:
        """Predict labels for a set of texts.

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
        """Predict probabilities for each class.

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
        X: Sequence[str],
        y: LabelType,
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        min_epochs: int | None = None,
        max_epochs: int | None = -1,
        early_stopping_patience: int | None = 5,
        test_size: float | int = 0.1,
        device: str = "auto",
        X_val: Sequence[str] | None = None,
        y_val: LabelType | None = None,
        class_weight: Literal["balanced"] | dict[str, float] | torch.Tensor | None = None,
        validation_steps: int | None = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
        token_dropout: float = 0.0,
    ) -> StaticModelForClassification:
        """Fit a model.

        This function trains the model with a plain torch training loop.
        It supports both single-label and multi-label classification.
        We use early stopping. After training, the weights of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        If `X_val` and `y_val` are not provided, the function will automatically
        split the training data into a train and validation set using `test_size`.

        The texts and labels are read and tokenized per batch. They can be lists, or columns of a Hugging Face
        dataset, such as `dataset["text"]`, which are not loaded into memory. The dataset must not have a transform.

        :param X: The texts to train on.
        :param y: The labels to train on. If the first element is a list, multi-label classification is assumed.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size. If None, a good batch size is chosen automatically.
        :param min_epochs: The minimum number of epochs to train for.
        :param max_epochs: The maximum number of epochs to train for.
            If this is -1, the model trains until early stopping is triggered.
        :param early_stopping_patience: The patience for early stopping.
            If this is None, early stopping is disabled.
        :param test_size: The size of the validation split if `X_val` is None: a fraction of the data, capped at
            10,000 rows, or a number of rows if it is an int. The split is stratified if `y` holds single
            labels.
        :param device: The device to train on. If this is "auto", the device is chosen automatically.
        :param X_val: The texts to be used for validation.
        :param y_val: The labels to be used for validation.
        :param class_weight: The weight of the classes. If None, all classes are weighted equally.
            If "balanced", weights are computed as the inverse class frequency.
            If a dict, it must map each class to its weight.
        :param validation_steps: The number of steps to run validation for. If None, validation steps are estimated from the data.
        :param random_seed: The random seed to use. Defaults to 42.
        :param token_dropout: The fraction of tokens to randomly drop from each training sample.
            Has no effect during validation. Must be in the range [0, 1).
        :return: The fitted model.
        """
        seed_everything(random_seed)
        logger.info("Re-initializing model.")
        self._check_inputs(X=X, y=y, X_val=X_val, y_val=y_val)

        label_counts = self._initialize_on_labels(y)
        if y_val is not None:
            self._check_validation_labels(y_val)
        self._initialize()
        resolved_class_weight = self._resolve_class_weight(class_weight, label_counts)
        train_dataset, val_dataset = self._create_datasets(
            X, y, X_val, y_val, test_size, stratify_by=None if self.multilabel else y
        )

        if self.multilabel:
            loss_function: nn.Module = nn.BCEWithLogitsLoss(pos_weight=resolved_class_weight)
            compute_metrics = _multilabel_classifier_metrics
        else:
            loss_function = nn.CrossEntropyLoss(weight=resolved_class_weight)
            compute_metrics = _classifier_metrics

        self._train(
            loss_function=loss_function,
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=self._determine_batch_size(batch_size, len(train_dataset)),
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            device=device,
            validation_steps=validation_steps,
            compute_metrics=compute_metrics,
            token_dropout=token_dropout,
        )

        return self

    def _resolve_class_weight(
        self,
        class_weight: Literal["balanced"] | dict[str, float] | torch.Tensor | None,
        counts: Mapping[Any, int],
    ) -> torch.Tensor | None:
        """Turn the `class_weight` passed to `fit` into a tensor with one weight per class.

        :param class_weight: The class weight passed to `fit`.
        :param counts: The number of times each class occurs.
        :return: The weight of each class, or None if `class_weight` is None.
        :raises ValueError: If `class_weight` is a tensor with the wrong length.
        """
        if class_weight is None:
            return None
        if isinstance(class_weight, torch.Tensor):
            logger.warning("You are passing a tensor as class weight. This will be removed in an upcoming version.")
            if len(class_weight) != len(self.classes_):
                raise ValueError("class_weight must have the same length as the number of classes.")
            class_weight = {self.classes_[idx]: w for idx, w in enumerate(class_weight.tolist())}
        return self._class_weight_from_counts(class_weight, counts)

    def _class_weight_from_counts(
        self, class_weight: dict[str, float] | Literal["balanced"], counts: Mapping[Any, int]
    ) -> torch.Tensor:
        """Determine the class weight for the classifier from the number of times each class occurs."""
        if class_weight == "balanced":
            total = sum(counts.values())
            n_classes = len(counts)
            # Reciprocal weight: upweight rare classes, downweight frequent ones
            weights = [total / (n_classes * counts[c]) for c in self.classes_]
        else:
            weights = [class_weight[c] for c in self.classes_]
        return torch.tensor(weights, dtype=torch.float32)

    def evaluate(
        self, X: list[str], y: LabelType, batch_size: int = 1024, threshold: float = 0.5
    ) -> dict[str, dict[str, float]]:
        """Evaluate the classifier on a given dataset.

        :param X: The texts to predict on.
        :param y: The ground truth labels.
        :param batch_size: The batch size.
        :param threshold: The threshold for multilabel classification.
        :return: A classification report, as a dictionary.
        """
        self.eval()
        predictions = self.predict(X, show_progress_bar=True, batch_size=batch_size, threshold=threshold)
        return evaluate_single_or_multi_label(predictions=predictions, y=y)

    def _initialize_on_labels(self, y: LabelType) -> Mapping[Any, int]:
        """Sets the output dimensionality and the classes from the labels.

        :param y: The labels. A column of a Hugging Face dataset is read in batches, without converting it to
            Python objects.
        :return: The number of times each class occurs.
        """
        self.multilabel, counts = _read_labels(y, "y")
        self.classes_ = sorted(counts)
        self.out_dim = len(self.classes_)
        return counts

    def _check_validation_labels(self, y_val: LabelType) -> None:
        """Check that the validation labels match the labels the classifier was initialized on.

        :param y_val: The validation labels.
        :raises ValueError: If `y_val` is multi-label and `y` is not, or the other way around, or if `y_val`
            contains classes that are not in `y`.
        """
        multilabel, counts = _read_labels(y_val, "y_val")
        if multilabel != self.multilabel:
            raise ValueError("y_val must be multi-label if and only if y is multi-label.")
        unknown = set(counts) - set(self.classes_)
        if unknown:
            raise ValueError(f"y_val contains classes that are not in y: {sorted(unknown, key=str)}.")

    def _to_targets(self, labels: Any) -> torch.Tensor:
        """Turn a batch of labels into targets.

        :param labels: The labels. A tensor or an array is converted to a list first.
        :return: The class indices, or the multi-hot vectors if the task is multilabel.
        :raises ValueError: If a label is not one of the classes.
        """
        if isinstance(labels, (torch.Tensor, np.ndarray)):
            labels = labels.tolist()
        index = {label: i for i, label in enumerate(self.classes_)}
        try:
            if not self.multilabel:
                return torch.tensor([index[label] for label in labels], dtype=torch.long)
            targets = torch.zeros(len(labels), len(index), dtype=torch.float)
            for row, sample_labels in enumerate(labels):
                if isinstance(sample_labels, (torch.Tensor, np.ndarray)):
                    sample_labels = sample_labels.tolist()
                targets[row, [index[label] for label in sample_labels]] = 1.0
            return targets
        except KeyError as error:
            raise ValueError(f"Label {error.args[0]!r} is not one of the classes {self.classes_}.") from None
