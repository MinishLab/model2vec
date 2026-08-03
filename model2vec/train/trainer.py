from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

MetricsFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, float]]

_UNBOUNDED_MAX_EPOCHS = 9999


def default_metrics(head_out: torch.Tensor, y: torch.Tensor, loss: torch.Tensor) -> dict[str, float]:
    """Validation metrics for tasks that only track loss (used for early stopping on val_loss)."""
    return {"val_loss": loss.item()}


class EarlyStopper:
    def __init__(self, mode: str, patience: int, min_delta: float = 0.001) -> None:
        """Track a monitored metric across validation checks and signal when it has stopped improving."""
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.wait = 0

    def step(self, value: float) -> bool:
        """Update with a new metric value, returning whether training should stop."""
        improved = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience


def _resolve_max_epochs(max_epochs: int | None) -> int:
    """Resolve `max_epochs` to a concrete cap, treating None or negative as unbounded."""
    if max_epochs is None or max_epochs < 0:
        return _UNBOUNDED_MAX_EPOCHS
    return max_epochs


def _step_plateau_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, latest_val_loss: float | None
) -> None:
    """Step a ReduceLROnPlateau scheduler with the latest val_loss, once per epoch, if one is available yet."""
    if latest_val_loss is not None:
        scheduler.step(latest_val_loss)


def resolve_device(device: str) -> torch.device:
    """Resolve a device string, with "auto" picking the best available accelerator."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _run_validation(
    model: nn.Module,
    loss_function: nn.Module,
    compute_metrics: MetricsFn,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    weighted_sums: dict[str, float] = defaultdict(float)
    total_samples = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        head_out = model(x)
        loss = loss_function(head_out, y)
        for key, value in compute_metrics(head_out, y, loss).items():
            weighted_sums[key] += value * batch_size
        total_samples += batch_size
    model.train()
    return {key: total / total_samples for key, total in weighted_sums.items()}


def run_training_loop(
    model: nn.Module,
    loss_function: nn.Module,
    learning_rate: float,
    val_metric: str,
    early_stopping_direction: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    early_stopping_patience: int | None,
    min_epochs: int | None,
    max_epochs: int | None,
    device: torch.device,
    val_check_interval: int | None,
    check_val_every_epoch: int | None,
    compute_metrics: MetricsFn = default_metrics,
) -> dict[str, torch.Tensor]:
    """Train `model` with a plain torch loop, validating and checkpointing on the configured cadence.

    :param model: The model to train, called as `head_out = model(x)`.
    :param loss_function: Computes the training and validation loss from `(head_out, y)`.
    :param learning_rate: The Adam learning rate.
    :param val_metric: The metric key (returned by `compute_metrics`) used for early stopping.
    :param early_stopping_direction: Either "min" or "max", the direction of improvement for `val_metric`.
    :param train_loader: The training data loader.
    :param val_loader: The validation data loader.
    :param early_stopping_patience: The number of validation checks without improvement before stopping.
        If None, early stopping is disabled.
    :param min_epochs: The minimum number of epochs to train for before early stopping can trigger.
    :param max_epochs: The maximum number of epochs to train for. If None or negative, unbounded.
    :param device: The device to train on.
    :param val_check_interval: If set, validate every this many training steps.
    :param check_val_every_epoch: If set, validate every this many epochs.
    :param compute_metrics: Computes validation metrics from `(head_out, y, loss)`. Defaults to just `val_loss`.
    :return: The model's state dict as of the last validation check before training stopped.
    """
    model.to(device)
    loss_function.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        threshold=0.03,
        threshold_mode="rel",
    )
    early_stopper = (
        EarlyStopper(mode=early_stopping_direction, patience=early_stopping_patience)
        if early_stopping_patience is not None
        else None
    )

    max_epochs = _resolve_max_epochs(max_epochs)

    last_checkpoint = copy.deepcopy(model.state_dict())
    current_epoch = 0
    global_step = 0
    postfix: dict[str, str] = {}
    latest_val_loss: float | None = None

    def validate_and_checkpoint() -> bool:
        nonlocal last_checkpoint, latest_val_loss
        metrics = _run_validation(model, loss_function, compute_metrics, val_loader, device)
        latest_val_loss = metrics["val_loss"]
        last_checkpoint = copy.deepcopy(model.state_dict())
        postfix.update({key: f"{value:.4f}" for key, value in metrics.items()})
        if early_stopper is None:
            return False
        return early_stopper.step(metrics[val_metric])

    while True:
        with tqdm(train_loader, desc=f"Epoch {current_epoch + 1}", unit="step") as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                head_out = model(x)
                loss = loss_function(head_out, y)
                loss.backward()
                optimizer.step()
                global_step += 1

                postfix["train_loss"] = f"{loss.item():.4f}"
                pbar.set_postfix(postfix)

                if val_check_interval is not None and global_step % val_check_interval == 0:
                    should_stop = validate_and_checkpoint()
                    pbar.set_postfix(postfix)
                    if should_stop and (min_epochs is None or current_epoch >= min_epochs):
                        return last_checkpoint

            current_epoch += 1

            if check_val_every_epoch is not None and current_epoch % check_val_every_epoch == 0:
                should_stop = validate_and_checkpoint()
                pbar.set_postfix(postfix)
                if should_stop and (min_epochs is None or current_epoch >= min_epochs):
                    return last_checkpoint

            _step_plateau_scheduler(scheduler, latest_val_loss)

            if current_epoch >= max_epochs:
                return last_checkpoint
