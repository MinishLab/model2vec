from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.utilities.types import OptimizerLRScheduler
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
        # Alias: Follows scikit-learn. Set to dummy classes
        self.classes_: list[str] = [str(x) for x in range(out_dim)]
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

    def predict(self, X: list[str]) -> list[str]:
        """Predict a class for a set of texts."""
        pred: list[str] = []
        for batch in range(0, len(X), 1024):
            logits = self._predict(X[batch : batch + 1024])
            pred.extend([self.classes[idx] for idx in logits.argmax(1)])

        return pred

    @torch.no_grad()
    def _predict(self, X: list[str]) -> torch.Tensor:
        input_ids = self.tokenize(X)
        vectors, _ = self.forward(input_ids)
        return vectors

    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Predict the probability of each class."""
        pred: list[np.ndarray] = []
        for batch in range(0, len(X), 1024):
            logits = self._predict(X[batch : batch + 1024])
            pred.append(torch.softmax(logits, dim=1).numpy())

        return np.concatenate(pred)

    def fit(
        self,
        X: list[str],
        y: list[str],
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        early_stopping_patience: int | None = 25,
        test_size: float = 0.1,
    ) -> ClassificationStaticModel:
        """Fit a model."""
        pl.seed_everything(42)
        self._initialize(y)

        train_texts, validation_texts, train_labels, validation_labels = self._train_test_split(
            X, y, test_size=test_size
        )

        train_dataset = self._prepare_dataset(train_texts, train_labels)
        val_dataset = self._prepare_dataset(validation_texts, validation_labels)

        c = ClassifierLightningModule(self, learning_rate=learning_rate)

        n_train_batches = len(train_dataset) // batch_size
        callbacks: list[Callback] = []
        if early_stopping_patience is not None:
            callback = EarlyStopping(monitor="val_accuracy", mode="max", patience=early_stopping_patience)
            callbacks.append(callback)

        if n_train_batches < 250:
            val_check_interval = None
            check_val_every_epoch = True
        else:
            val_check_interval = max(250, 2 * len(val_dataset) // batch_size)
            check_val_every_epoch = False
        trainer = pl.Trainer(
            max_epochs=500,
            callbacks=callbacks,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_epoch,
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

    def _initialize(self, y: list[str]) -> None:
        """Sets the out dimensionality, the classes and initializes the head."""
        classes = sorted(set(y))
        self.classes_ = classes

        if len(self.classes) != self.out_dim:
            self.out_dim = len(self.classes)

        self.head = self.construct_head()
        self.embeddings = nn.Embedding.from_pretrained(self.vectors.clone(), freeze=False, padding_idx=self.pad_id)
        self.train()

    def _prepare_dataset(self, X: list[str], y: list[str]) -> TextDataset:
        """Prepare a dataset."""
        tokenized: list[list[int]] = [
            encoding.ids for encoding in self.tokenizer.encode_batch_fast(X, add_special_tokens=False)
        ]
        labels_tensor = torch.Tensor([self.classes.index(label) for label in y]).long()
        return TextDataset(tokenized, labels_tensor)

    def _train_test_split(
        self, X: list[str], y: list[str], test_size: float
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Split the data."""
        label_counts = Counter(y)
        if min(label_counts.values()) < 2:
            logger.info("Some classes have less than 2 samples. Stratification is disabled.")
            return train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
        return train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True, stratify=y)


class ClassifierLightningModule(pl.LightningModule):
    def __init__(self, model: ClassificationStaticModel, learning_rate: float) -> None:
        """Initialize the lightningmodule."""
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Simple Adam optimizer."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6,
            threshold=0.03,
            threshold_mode="rel",
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
