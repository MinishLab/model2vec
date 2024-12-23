from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from model2vec.train.base import FinetunableStaticModel, TextDataset

logger = logging.getLogger(__name__)

# Try to import wandb for logging
try:
    import wandb

    _WANDB_INSTALLED = True
except ImportError:
    _WANDB_INSTALLED = False


def _init_wandb(project_name: str, config: dict | None = None) -> None:
    """Initialize Weights & Biases for tracking experiments if wandb is installed."""
    if _WANDB_INSTALLED:
        wandb.init(project=project_name, config=config)
        logger.info(f"W&B initialized with project: {project_name}")
    else:
        logger.info("Skipping W&B initialization since wandb is not installed.")


def train_supervised(  # noqa: C901
    model: FinetunableStaticModel,
    train_dataset: TextDataset,
    validation_dataset: TextDataset | None,
    loss_calculator: Callable,
    max_epochs: int = 50,
    min_epochs: int = 1,
    patience: int | None = 5,
    patience_min_delta: float = 0.001,
    batch_size: int = 256,
    wandb_project_name: str | None = None,
    wandb_config: dict | None = None,
    lr_scheduler_patience: int = 3,
    lr_scheduler_min_delta: float = 0.03,
    lr_model: float = 0.001,
    lr_head: float = 0.001,
) -> FinetunableStaticModel:
    """
    Train a StaticModel.

    :param model: The model to train.
    :param train_dataset: The training dataset.
    :param validation_dataset: The validation dataset. If this is None and patience is set,
        early stopping is performed on the training set.
    :param loss_calculator: A function that calculates the loss.
    :param max_epochs: The maximum number of epochs to train.
    :param min_epochs: The minimum number of epochs to train.
    :param patience: The number of epochs to wait before early stopping.
    :param patience_min_delta: The minimum delta for early stopping.
    :param batch_size: The batch size.
    :param wandb_project_name: The name of the project for W&B.
    :param wandb_config: The configuration for W&B.
    :param lr_scheduler_patience: The patience for the learning rate scheduler.
    :param lr_scheduler_min_delta: The minimum delta for the learning rate scheduler.
    :param lr_model: The learning rate for the model.
    :param lr_head: The learning rate for the head.
    :return: The trained model.
    """
    if wandb_config is None:
        wandb_config = {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "min_epochs": min_epochs,
            "patience": patience,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_min_delta": lr_scheduler_min_delta,
            "lr_model": lr_model,
            "lr_linear": lr_head,
        }

    # Initialize W&B only if wandb is installed and project name is provided
    if _WANDB_INSTALLED and wandb_project_name:
        _init_wandb(project_name=wandb_project_name, config=wandb_config)
        wandb_initialized = True
    else:
        wandb_initialized = False

    train_dataloader = train_dataset.to_dataloader(shuffle=True, batch_size=batch_size)
    if validation_dataset is not None:
        validation_dataloader = validation_dataset.to_dataloader(shuffle=False, batch_size=batch_size)
    else:
        validation_dataloader = None

    # Separate parameters for model and linear layer
    model_params = list(model.embeddings.parameters()) + [model.w]
    head_params = model.head.parameters()

    # Create optimizer with separate parameter groups
    optimizer = torch.optim.Adam([{"params": model_params, "lr": lr_model}, {"params": head_params, "lr": lr_head}])

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_scheduler_patience,
        verbose=True,
        min_lr=1e-6,
        threshold=lr_scheduler_min_delta,
        threshold_mode="rel",
    )

    lowest_loss = float("inf")
    param_dict = model.state_dict()
    curr_patience = patience

    try:
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch}")
            model.train()

            # Track train loss separately
            tracked_losses = defaultdict(list)
            barred_train = tqdm(train_dataloader, desc=f"Epoch {epoch:03d} [Train]")

            for x, y in barred_train:
                optimizer.zero_grad()
                x = x.to(model.device)
                head_out, emb_out = model(x)
                losses: dict[str, torch.Tensor] = loss_calculator(head_out, emb_out, y)

                train_loss = losses["loss"]
                train_loss.backward()

                optimizer.step()

                for loss_name, loss in losses.items():
                    tracked_losses[loss_name].append(loss.item())

                barred_train.set_description_str(f"Train Loss: {np.mean(tracked_losses['loss'][-10:]):.3f}")

            # Calculate average losses
            avg_train_loss = float(np.mean(tracked_losses["loss"]))

            # Step the scheduler with the current training loss
            scheduler.step(avg_train_loss)

            # Get current learning rates
            current_lr_model = optimizer.param_groups[0]["lr"]
            current_lr_linear = optimizer.param_groups[1]["lr"]

            # Log training loss and learning rates to wandb
            if wandb_initialized:
                wandb.log(
                    {
                        "epoch": epoch,
                        "learning_rate_model": current_lr_model,
                        "learning_rate_linear": current_lr_linear,
                        **{k: np.mean(v) for k, v in tracked_losses.items()},
                    }
                )

            logger.info(f"Training loss: {avg_train_loss:.3f}")

            patience_loss = avg_train_loss
            if validation_dataloader is not None:
                model.eval()
                avg_validation_loss: list[float] = []
                for x, y in validation_dataloader:
                    optimizer.zero_grad()
                    x = x.to(model.device)
                    head_out, emb_out = model(x)
                    losses = loss_calculator(head_out, emb_out, y)

                    val_loss: float = losses["loss"].item()
                    avg_validation_loss.append(val_loss)

                patience_loss = float(np.mean(avg_validation_loss))
                logger.info(f"Validation loss: {patience_loss:.3f}")

            # Early stopping logic based on training loss
            curr_patience, lowest_loss = _track_patience(
                patience, curr_patience, epoch, min_epochs, patience_min_delta, patience_loss, lowest_loss
            )
            if curr_patience is not None and curr_patience == 0:
                break

            model.train()

    except KeyboardInterrupt:
        logger.info("Training interrupted")

    model.eval()
    # Load best model based on training loss
    model.load_state_dict(param_dict)

    return model


def _track_patience(
    patience: None | int,
    curr_patience: int | None,
    epoch: int,
    min_epochs: int,
    patience_min_delta: float,
    curr_loss: float,
    lowest_loss: float,
) -> tuple[int | None, float]:
    if patience is not None and curr_patience is not None and epoch >= min_epochs:
        patience_str = "🌝" * curr_patience
        logger.info(f"Patience level: {patience_str}")
        logger.info(f"Lowest train loss: {lowest_loss:.3f}")
        if (lowest_loss - curr_loss) > patience_min_delta:
            curr_patience = patience
            lowest_loss = curr_loss
            return patience, lowest_loss
        else:
            curr_patience -= 1
            return curr_patience, lowest_loss
    else:
        # We shouldn't ever stop
        return patience, float("inf")
