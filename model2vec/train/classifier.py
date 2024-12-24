from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn

from model2vec.train.base import FinetunableStaticModel, TextDataset
from model2vec.train.train_loop import train_supervised


class ClassificationStaticModel(FinetunableStaticModel):
    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        n_layers: int,
        hidden_dim: int,
        out_dim: int,
        pad_id: int = 0,
    ) -> None:
        """Initialize a standard classifier model."""
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # Alias: Follows scikit-learn.
        self.classes_: list[str] = []
        super().__init__(vectors=vectors, out_dim=out_dim, pad_id=pad_id, tokenizer=tokenizer)

    @property
    def classes(self) -> list[str]:
        """Return all clasess in the correct order."""
        return self.classes_

    def construct_head(self) -> nn.Module:
        """Constructs a simple classifier head."""
        if self.n_layers == 0:
            return nn.Linear(self.embed_dim, self.out_dim)
        modules = [nn.Linear(self.embed_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_layers - 1):
            modules.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        modules.append(nn.Linear(self.hidden_dim, self.out_dim))

        return nn.Sequential(*modules)

    def predict(self, texts: list[str]) -> list[str]:
        """Predict a class for a set of texts."""
        logits = self._predict(texts)

        return [self.classes[idx] for idx in logits.argmax(1)]

    @torch.no_grad()
    def _predict(self, texts: list[str]) -> torch.Tensor:
        input_ids = self.tokenize(texts)
        vectors, _ = self.forward(input_ids)
        return vectors

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict the probability of each class."""
        logits = self._predict(texts)

        return torch.softmax(logits, dim=1).numpy()

    def loss_calculator(
        self, head_out: torch.Tensor, embedding_out: torch.Tensor, y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Calculates the loss for this specific task."""
        # Separate loss components
        loss = nn.functional.cross_entropy(head_out, y.to(self.device)).mean()
        return {"loss": loss}

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[str],
        validation_texts: list[str],
        validation_labels: list[str],
        **kwargs: Any,
    ) -> FinetunableStaticModel:
        """Fit a model."""
        classes = sorted(set(train_labels) | set(validation_labels))
        self.classes_ = classes

        if len(self.classes) != self.out_dim:
            self.out_dim = len(self.classes)
            self.head = self.construct_head()

        label_mapping = {label: idx for idx, label in enumerate(self.classes)}
        # Turn labels into a LongTensor
        train_labels_tensor = torch.Tensor([label_mapping[label] for label in train_labels]).long()
        train_dataset = TextDataset(train_texts, train_labels_tensor, self.tokenizer)
        val_labels_tensor = torch.Tensor([label_mapping[label] for label in validation_labels]).long()
        val_dataset = TextDataset(validation_texts, val_labels_tensor, self.tokenizer)

        return train_supervised(self, train_dataset, val_dataset, self.loss_calculator, **kwargs)
