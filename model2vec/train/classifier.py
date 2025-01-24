from __future__ import annotations

import logging
from collections import Counter
from tempfile import TemporaryDirectory

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from tokenizers import Tokenizer
from torch import nn
from tqdm import trange

from model2vec.inference import StaticModelPipeline
from model2vec.train.base import FinetunableStaticModel, TextDataset

logger = logging.getLogger(__name__)


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
        super().__init__(vectors=vectors, out_dim=out_dim, pad_id=pad_id, tokenizer=tokenizer)

    @property
    def classes(self) -> list[str]:
        """Return all clasess in the correct order."""
        return self.classes_

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

    def predict(self, X: list[str], show_progress_bar: bool = False, batch_size: int = 1024) -> np.ndarray:
        """Predict a class for a set of texts."""
        pred: list[str] = []
        for batch in trange(0, len(X), batch_size, disable=not show_progress_bar):
            logits = self._predict_single_batch(X[batch : batch + batch_size])
            pred.extend([self.classes[idx] for idx in logits.argmax(1)])

        return np.asarray(pred)

    @torch.no_grad()
    def _predict_single_batch(self, X: list[str]) -> torch.Tensor:
        input_ids = self.tokenize(X)
        vectors, _ = self.forward(input_ids)
        return vectors

    def predict_proba(self, X: list[str], show_progress_bar: bool = False, batch_size: int = 1024) -> np.ndarray:
        """Predict the probability of each class."""
        pred: list[np.ndarray] = []
        for batch in trange(0, len(X), batch_size, disable=not show_progress_bar):
            logits = self._predict_single_batch(X[batch : batch + batch_size])
            pred.append(torch.softmax(logits, dim=1).numpy())

        return np.concatenate(pred)

    def fit(
        self,
        X: list[str],
        y: list[str],
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
    ) -> StaticModelForClassification:
        """
        Fit a model.

        This function creates a Lightning Trainer object and fits the model to the data.
        We use early stopping. After training, the weigths of the best model are loaded back into the model.

        This function seeds everything with a seed of 42, so the results are reproducible.
        It also splits the data into a train and validation set, again with a random seed.

        :param X: The texts to train on.
        :param y: The labels to train on.
        :param learning_rate: The learning rate.
        :param batch_size: The batch size.
            If this is None, a good batch size is chosen automatically.
        :param early_stopping_patience: The patience for early stopping.
            If this is None, early stopping is disabled.
        :param test_size: The test size for the train-test split.
        :param device: The device to train on. If this is "auto", the device is chosen automatically.
        :return: The fitted model.
        """
        pl.seed_everything(42)
        logger.info("Re-initializing model.")
        self._initialize(y)

        train_texts, validation_texts, train_labels, validation_labels = self._train_test_split(
            X, y, test_size=test_size
        )

        if batch_size is None:
            batch_size = max(min(32, len(train_texts) // 10), 512)
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
                max_epochs=500,
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

    def _initialize(self, y: list[str]) -> None:
        """Sets the out dimensionality, the classes and initializes the head."""
        classes = sorted(set(y))
        self.classes_ = classes

        if len(self.classes) != self.out_dim:
            self.out_dim = len(self.classes)

        self.head = self.construct_head()
        self.embeddings = nn.Embedding.from_pretrained(self.vectors.clone(), freeze=False, padding_idx=self.pad_id)
        self.w = self.construct_weights()
        self.train()

    def _prepare_dataset(self, X: list[str], y: list[str], max_length: int = 512) -> TextDataset:
        """Prepare a dataset."""
        # This is a speed optimization.
        # assumes a mean token length of 10, which is really high, so safe.
        truncate_length = max_length * 10
        X = [x[:truncate_length] for x in X]
        tokenized: list[list[int]] = [
            encoding.ids[:max_length] for encoding in self.tokenizer.encode_batch_fast(X, add_special_tokens=False)
        ]
        labels_tensor = torch.Tensor([self.classes.index(label) for label in y]).long()

        return TextDataset(tokenized, labels_tensor)

    @staticmethod
    def _train_test_split(
        X: list[str], y: list[str], test_size: float
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Split the data."""
        label_counts = Counter(y)
        if min(label_counts.values()) < 2:
            logger.info("Some classes have less than 2 samples. Stratification is disabled.")
            return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
        return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y)

    def to_pipeline(self) -> StaticModelPipeline:
        """Convert the model to an sklearn pipeline."""
        static_model = self.to_static_model()

        random_state = np.random.RandomState(42)
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
        # Set to softmax
        mlp_head.out_activation_ = "softmax"

        return StaticModelPipeline(static_model, converted)


class _ClassifierLightningModule(pl.LightningModule):
    def __init__(self, model: StaticModelForClassification, learning_rate: float) -> None:
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
