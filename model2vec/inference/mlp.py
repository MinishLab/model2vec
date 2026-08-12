from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Activation(str, Enum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    IDENTITY = "identity"


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - x.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


@dataclass
class Layer:
    weight: np.ndarray
    bias: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the linear transformation."""
        return x @ self.weight.T + self.bias


class MLPHead:
    def __init__(
        self,
        layers: list[Layer],
        activation: Activation,
        classes: np.ndarray | None = None,
    ) -> None:
        """An MLP with ReLU activation.

        :param layers: The linear layers, in order.
        :param activation: The output activation.
        :param classes: The classes, if the task is a classification task.
        """
        self.layers = layers
        self.activation = activation
        self.classes_ = classes

    def _logits(self, X: np.ndarray) -> np.ndarray:
        """Run the forward through the layers."""
        out = X
        *hidden_layers, last_layer = self.layers
        for layer in hidden_layers:
            out = np.maximum(layer(out), 0.0)
        return last_layer(out)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities, applying the output activation to the raw logits."""
        logits = self._logits(X)
        match self.activation:
            case Activation.SOFTMAX:
                return _softmax(logits)
            case Activation.SIGMOID:
                return _sigmoid(logits)
            case Activation.IDENTITY:
                return logits

    def predict_index(self, X: np.ndarray) -> np.ndarray:
        """Predict the index of the most likely class."""
        return self._logits(X).argmax(axis=1)

    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        """Predict the raw (identity-activation) output, e.g. for a projector head."""
        return self._logits(X)
