import logging
from pathlib import Path
from typing import Any, Optional, Type

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from tokenizers import Tokenizer
from torch import nn

from model2vec import StaticModel
from model2vec.knowledge_distill.utils import calculate_token_probabilities, collect_means_and_texts
from model2vec.train.base import FinetunableStaticModel, ModelType, TextDataset

logger = logging.getLogger(__name__)


class KnowledgeDistillationDataset(TextDataset):
    """Dataset class for Knowledge Distillation training."""

    def __init__(self, texts: list[str], targets: torch.Tensor, tokenizer: Tokenizer) -> None:
        """Initialize a Knowledge Distillation dataset."""
        tokenized_texts = [encoding.ids for encoding in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
        super().__init__(tokenized_texts, targets)


class KnowledgeDistillationModel(FinetunableStaticModel, pl.LightningModule):
    """A knowledge distillation model that learns to map token embeddings to target vectors."""

    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        out_dim: int,
        pad_id: int = 0,
        lr_model: float = 3e-3,
        lr_linear: float = 1e-2,
        cosine_weight: float = 1.0,
        mse_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a TokenlearnModel.

        :param vectors: (vocab_size, embed_dim) float tensor
        :param tokenizer: the tokenizer
        :param out_dim: dimension of the linear head output
        :param pad_id: the padding ID
        :param lr_model: learning rate for the embeddings + weighting param
        :param lr_linear: learning rate for the linear head
        :param cosine_weight: weight for the (1 - cosSim) loss
        :param mse_weight: weight for the MSE penalty on the mean embedding
        :param **kwargs: Additional keyword arguments passed to FinetunableStaticModel.
        """
        super().__init__(vectors=vectors, tokenizer=tokenizer, out_dim=out_dim, pad_id=pad_id, **kwargs)
        self.lr_model = lr_model
        self.lr_linear = lr_linear
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.w = self.construct_weights()

    def construct_weights(self) -> nn.Parameter:
        """Construct the weights for the model."""
        weights = torch.ones(len(self.vectors))  # Change from zeros to ones
        weights[self.pad_id] = 0  # Make sure padding gets ignored
        return nn.Parameter(weights)

    def sub_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mean pooling."""
        w = self.w[x]
        zeros = (x != self.pad_id).float()
        length = zeros.sum(1)
        embedded = self.embeddings(x)

        # Zero out the padding
        embedded = embedded * zeros[:, :, None]
        embedded = (embedded * w[:, :, None]).sum(1) / (w.sum(1)[:, None])

        embedded = embedded / length[:, None]

        return embedded

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean pooling, and a classifier layer after."""
        encoded = self.sub_forward(input_ids)
        return self.head(encoded), encoded

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """The training step for the model."""
        input_ids, target_vectors = batch
        y_hat, mean_emb = self.forward(input_ids)
        cos_loss = 1.0 - F.cosine_similarity(y_hat, target_vectors, dim=1).mean()
        mse_loss = (mean_emb**2).mean()
        loss = self.cosine_weight * cos_loss + self.mse_weight * mse_loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure the optimizer for the model."""
        model_params = list(self.embeddings.parameters()) + [self.w]
        linear_params = self.head.parameters()
        optimizer = torch.optim.Adam(
            [
                {"params": model_params, "lr": self.lr_model},
                {"params": linear_params, "lr": self.lr_linear},
            ]
        )
        return optimizer

    @classmethod
    def from_static_model(
        cls: Type[ModelType],
        model: StaticModel,
        out_dim: Optional[int] = None,
        lr_model: float = 3e-3,
        lr_linear: float = 1e-2,
        cosine_weight: float = 1.0,
        mse_weight: float = 1.0,
        **kwargs: Any,
    ) -> ModelType:
        """Method to wrap a StaticModel in a TokenlearnModel."""
        if out_dim is None:
            out_dim = model.dim
        embeddings = torch.from_numpy(np.nan_to_num(model.embedding))
        return cls(
            vectors=embeddings,
            tokenizer=model.tokenizer,
            out_dim=out_dim,
            pad_id=model.tokenizer.token_to_id("[PAD]"),
            lr_model=lr_model,
            lr_linear=lr_linear,
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
            **kwargs,
        )

    def fit(
        self,
        dataset: TextDataset,
        batch_size: int = 256,
        max_epochs: int = 50,
        device: str = "cpu",
        patience: int | None = 5,
    ) -> None:
        """Fit the model."""
        callbacks: list[pl.Callback] = []
        if patience is not None:
            callbacks.append(EarlyStopping(monitor="train_loss", mode="min", patience=patience))

        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            dirpath="checkpoints/",
            filename="best_model",
        )
        callbacks.append(checkpoint_callback)

        train_loader = dataset.to_dataloader(batch_size=batch_size, shuffle=True)
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, callbacks=callbacks)
        trainer.fit(self, train_loader)

        # Load the best checkpoint after training
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"Loading best model from {best_model_path}")
            self.load_state_dict(torch.load(best_model_path)["state_dict"])

    def apply_weighting(self, texts: list[str], alpha: float = 1e-3, pca_dims: int = 256) -> StaticModel:
        """
        Post-training step to apply SIF weighting + PCA to the embeddings.

        :param texts: The corpus on which to compute frequencies.
        :param alpha: The SIF alpha.
        :param pca_dims: The dimensionality to keep after PCA.
        :return: A new StaticModel with weighted embeddings.
        """
        logger.info("Applying SIF weighting + PCA to the embeddings.")

        # Compute token frequencies
        probas = calculate_token_probabilities(self.tokenizer, texts)

        # Move embeddings to CPU and convert to NumPy
        with torch.no_grad():
            embeddings_weight = self.embeddings.weight.detach().cpu().numpy()
            embeddings_weight = np.nan_to_num(embeddings_weight)

            # Apply PCA
            p = PCA(n_components=pca_dims)
            emb_pca = p.fit_transform(embeddings_weight)

            # Apply SIF weighting
            sif_weights = alpha / (alpha + probas)
            emb_pca *= sif_weights[:, None]

        # Create a new StaticModel with the modified embeddings
        weighted_model = StaticModel(emb_pca, self.tokenizer, normalize=True)

        return weighted_model

    def save_pretrained(self, save_directory: str) -> None:
        """Convert the trained model to a StaticModel and save it."""
        final_static = self.to_static_model()
        final_static.save_pretrained(save_directory)
        logger.info(f"Saved TokenlearnModel as a static model to '{save_directory}'")


def main() -> None:
    """Run the tokenlearn training script."""
    # Initialize a StaticModel
    model = StaticModel.from_pretrained("minishlab/M2V_base_output")
    model.normalize = True

    # Collect paths for training data
    paths = sorted(Path("../tokenlearn/data/c4_features_bgebase_test").glob("*.json"))
    X, y = collect_means_and_texts(paths)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert `y` to a tensor and move it to device
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # Convert to TokenlearnDataset
    dataset = KnowledgeDistillationDataset(X, y_tensor, tokenizer=model.tokenizer)

    # Create a TokenlearnModel from the StaticModel
    tokenlearn_model = KnowledgeDistillationModel.from_static_model(model, out_dim=y_tensor.shape[1])
    tokenlearn_model.to(device)

    # Fit the model
    tokenlearn_model.fit(dataset, batch_size=256, max_epochs=50, device=device)

    # Apply SIF weighting + PCA to the embeddings
    tokenlearn_model.apply_weighting(X, alpha=1e-3, pca_dims=256)

    # Save the final static model
    tokenlearn_model.save_pretrained("models/potion-base-8M-reproduce-v1")


if __name__ == "__main__":
    main()
