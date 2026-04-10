from typing import cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.metrics import jaccard_score
from torch import nn


class StaticLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float) -> None:
        """Initialize the LightningModule."""
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = self.cosine_distance

    def cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns the cosine distance loss function."""
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return (1 - torch.sum(x * y, dim=1)).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step using cross-entropy loss for single-label and binary cross-entropy for multilabel training."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step computing loss and accuracy."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        self.log("val_loss", loss, prog_bar=True)

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


class ClassifierLightningModule(StaticLightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, class_weight: torch.Tensor | None = None) -> None:
        """Initialize the LightningModule."""
        super().__init__(model, learning_rate)
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss(weight=class_weight)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step computing loss and accuracy."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        accuracy = (head_out.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss


class MultiLabelClassifierLightningModule(StaticLightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, class_weight: torch.Tensor | None = None) -> None:
        """Initialize the LightningModule."""
        super().__init__(model, learning_rate)
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weight)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step computing loss and accuracy."""
        x, y = batch
        head_out, _ = self.model(x)
        loss = self.loss_function(head_out, y)
        preds = (torch.sigmoid(head_out) > 0.5).float()
        # Multilabel accuracy is defined as the Jaccard score averaged over samples.
        accuracy = cast(float, jaccard_score(y.cpu(), preds.cpu(), average="samples"))
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss
