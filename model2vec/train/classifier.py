from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch import nn

from model2vec.train.base import FinetunableStaticModel, TextDataset

logger = logging.getLogger(__name__)


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
        modules = [
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(self.n_layers - 1):
            modules.extend(
                [nn.Dropout(0.5), nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.ReLU()]
            )
        modules.extend([nn.Linear(self.hidden_dim, self.out_dim)])

        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

        return nn.Sequential(*modules)

    def predict(self, texts: list[str]) -> list[str]:
        """Predict a class for a set of texts."""
        pred: list[str] = []
        for batch in range(0, len(texts), 1024):
            logits = self._predict(texts[batch : batch + 1024])
            pred.extend([self.classes[idx] for idx in logits.argmax(1)])

        return pred

    @torch.no_grad()
    def _predict(self, texts: list[str]) -> torch.Tensor:
        input_ids = self.tokenize(texts)
        vectors, _ = self.forward(input_ids)
        return vectors

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict the probability of each class."""
        pred: list[np.ndarray] = []
        for batch in range(0, len(texts), 1024):
            logits = self._predict(texts[batch : batch + 1024])
            pred.append(torch.softmax(logits, dim=1).numpy())

        return np.concatenate(pred)

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        **kwargs: Any,
    ) -> ClassificationStaticModel:
        """Fit a model."""
        pl.seed_everything(42)
        classes = sorted(set(labels))
        self.classes_ = classes

        if len(self.classes) != self.out_dim:
            self.out_dim = len(self.classes)

        self.head = self.construct_head()
        self.embeddings = nn.Embedding.from_pretrained(self.vectors.clone(), freeze=False, padding_idx=self.pad_id)

        label_mapping = {label: idx for idx, label in enumerate(self.classes)}
        label_counts = Counter(labels)
        if min(label_counts.values()) < 2:
            logger.info("Some classes have less than 2 samples. Stratification is disabled.")
            train_texts, validation_texts, train_labels, validation_labels = train_test_split(
                texts, labels, test_size=0.1, random_state=42, shuffle=True
            )
        else:
            train_texts, validation_texts, train_labels, validation_labels = train_test_split(
                texts, labels, test_size=0.1, random_state=42, shuffle=True, stratify=labels
            )

        # Turn labels into a LongTensor
        train_tokenized: list[list[int]] = [
            encoding.ids for encoding in self.tokenizer.encode_batch_fast(train_texts, add_special_tokens=False)
        ]
        train_labels_tensor = torch.Tensor([label_mapping[label] for label in train_labels]).long()
        train_dataset = TextDataset(train_tokenized, train_labels_tensor)

        val_tokenized: list[list[int]] = [
            encoding.ids for encoding in self.tokenizer.encode_batch_fast(validation_texts, add_special_tokens=False)
        ]
        val_labels_tensor = torch.Tensor([label_mapping[label] for label in validation_labels]).long()
        val_dataset = TextDataset(val_tokenized, val_labels_tensor)

        c = ClassifierLightningModule(self)

        batch_size = 32
        n_train_batches = len(train_dataset) // batch_size
        callbacks: list[Callback] = [EarlyStopping(monitor="val_accuracy", mode="max", patience=5)]
        if n_train_batches < 250:
            trainer = pl.Trainer(max_epochs=500, callbacks=callbacks, check_val_every_n_epoch=1)
        else:
            val_check_interval = max(250, 2 * len(val_dataset) // batch_size)
            trainer = pl.Trainer(
                max_epochs=500, callbacks=callbacks, val_check_interval=val_check_interval, check_val_every_n_epoch=None
            )

        trainer.fit(
            c,
            train_dataloaders=train_dataset.to_dataloader(shuffle=True, batch_size=batch_size),
            val_dataloaders=val_dataset.to_dataloader(shuffle=False, batch_size=batch_size),
        )
        best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore

        state_dict = {
            k.removeprefix("model."): v for k, v in torch.load(best_model_path, weights_only=True)["state_dict"].items()
        }
        self.load_state_dict(state_dict)

        self.eval()

        return self


class ClassifierLightningModule(pl.LightningModule):
    def __init__(self, model: ClassificationStaticModel) -> None:
        """Initialize the lightningmodule."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Simple training step using cross entropy loss."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = nn.functional.cross_entropy(head_out, y).mean()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Simple validation step using cross entropy loss and accuracy."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = nn.functional.cross_entropy(head_out, y).mean()
        accuracy = (head_out.argmax(1) == y).float().mean()

        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Simple Adam optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
