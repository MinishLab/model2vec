import logging
from typing import Any, List, Optional, Type

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model2vec import StaticModel
from model2vec.train.base import FinetunableStaticModel, ModelType, TextDataset

logger = logging.getLogger(__name__)
from pathlib import Path

from tokenizers import Tokenizer

from model2vec.train.utils import calculate_token_probabilities, collect_means_and_texts


class TokenlearnDataset(TextDataset):
    """Dataset class for Tokenlearn training."""

    def __init__(self, texts: List[str], targets: torch.Tensor, tokenizer: Tokenizer) -> None:
        """Initialize a TokenlearnDataset."""
        tokenized_texts = [encoding.ids for encoding in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
        super().__init__(tokenized_texts, targets)


class TokenlearnModel(FinetunableStaticModel, pl.LightningModule):
    """A TokenLearn model that learns to map token embeddings to target vectors."""

    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer: Tokenizer,
        out_dim: int = 2,
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

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """The training step for the model."""
        input_ids, target_vectors = batch
        y_hat, mean_emb = self.forward(input_ids)  # calls FinetunableStaticModel.forward
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
        batch_size: int = 32,
        max_epochs: int = 5,
        device: str = "cpu",
        patience: Optional[int] = None,
    ) -> None:
        """Fit the model."""
        callbacks: list[pl.Callback] = []
        if patience is not None:
            callbacks.append(EarlyStopping(monitor="train_loss", mode="min", patience=patience))
        train_loader = dataset.to_dataloader(batch_size=batch_size, shuffle=True)
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, callbacks=callbacks)
        trainer.fit(self, train_loader)

    def apply_weighting(self, texts: list[str], alpha: float = 1e-3, pca_dims: int = 256) -> None:
        """
        Post-training step to apply SIF weighting + PCA to the embeddings.

        Does the following:
        1) Count token frequencies across `texts`.
        2) PCA transform embeddings to `pca_dims`.
        3) Weight each token by alpha / (alpha + p(token)).
        4) Place the new embeddings back into self.embeddings.weight.

        :param texts: The corpus on which to compute frequencies.
        :param alpha: The SIF alpha.
        :param pca_dims: The dimensionality to keep after PCA.
        """
        logger.info("Applying SIF weighting + PCA to the embeddings.")
        # 1) freq distribution
        probas = calculate_token_probabilities(self.tokenizer, texts)

        # 2) fetch current embeddings as np array
        current_emb = self.embeddings.weight.detach().cpu().numpy()
        current_emb = np.nan_to_num(current_emb)

        # 3) PCA to `pca_dims`
        p = PCA(n_components=pca_dims)
        emb_pca = p.fit_transform(current_emb)  # shape: [vocab_size, pca_dims]

        # 4) SIF weighting
        # alpha / (alpha + freq)
        f = alpha / (alpha + probas)
        emb_pca *= f[:, None]

        # 5) store back into self.embeddings.weight
        with torch.no_grad():
            new_emb = torch.from_numpy(emb_pca).float()
            if new_emb.shape[1] == self.embed_dim:
                # If pca_dims == self.embed_dim, we can store back directly.
                self.embeddings.weight.copy_(new_emb.to(self.embeddings.weight.device))
            else:
                # If pca_dims != embed_dim,
                # we must either resize self.head, or store it but break the shape.
                # Typically you'd want pca_dims == self.embed_dim for a direct overwrite.
                logger.warning(
                    f"PCA dims={pca_dims} != embed_dim={self.embed_dim}."
                    " You have changed the embedding dimension. The linear head may need updating."
                )
                # For simplicity, let's just replace and assume user knows they changed dim.
                self.embeddings = nn.Embedding.from_pretrained(new_emb, freeze=False, padding_idx=self.pad_id)
                self.embed_dim = pca_dims
                # Optionally also re-init self.head if you want out_dim <-> embed_dim consistent.

        logger.info(f"SIF weighting + PCA complete. New embedding shape: {self.embeddings.weight.shape}.")

    def save_pretrained(self, save_directory: str) -> None:
        """Convert the trained model to a StaticModel and save it."""
        final_static = self.to_static_model()
        final_static.save_pretrained(save_directory)
        logger.info(f"Saved TokenlearnModel as a static model to '{save_directory}'")


def run() -> None:
    """Run the tokenlearn training script."""
    # Initialize a StaticModel
    model = StaticModel.from_pretrained("minishlab/M2V_base_output")

    # Collect paths for training data
    paths = sorted(Path("../tokenlearn/data/c4_features_bgebase_test").glob("*.json"))
    X, y = collect_means_and_texts(paths)

    # Convert `y` to a tensor with correct dtype
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Convert to TokenlearnDataset
    dataset = TokenlearnDataset(X, y_tensor, tokenizer=model.tokenizer)

    # Wrap it in our TokenlearnModel
    tokenlearn_model = TokenlearnModel.from_static_model(model, out_dim=y_tensor.shape[1])

    # Fit the model
    tokenlearn_model.fit(dataset, batch_size=8, max_epochs=1)
    tokenlearn_model.apply_weighting(X, alpha=1e-3, pca_dims=256)
    # Save the final static model
    tokenlearn_model.save_pretrained("local/tokenlearn_test")


if __name__ == "__main__":
    run()


# import logging
# from typing import List, Optional

# import lightning as pl
# import numpy as np
# import torch
# import torch.nn.functional as F
# from lightning.pytorch.callbacks import EarlyStopping
# from sklearn.decomposition import PCA
# from torch import nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from model2vec import StaticModel
# from model2vec.train.base import FinetunableStaticModel, ModelType, TextDataset

# logger = logging.getLogger(__name__)
# from collections import Counter
# from pathlib import Path

# from more_itertools import batched
# from tokenizers import Tokenizer

# from model2vec.train.utils import calculate_token_probabilities, collect_means_and_texts


# class TokenlearnDataset(TextDataset):
#     def __init__(self, texts: list[str], targets: torch.Tensor, tokenizer: Tokenizer) -> None:
#         """Initialize a TokenlearnDataset."""
#         tokenized_texts = [encoding.ids for encoding in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
#         super().__init__(tokenized_texts, targets)


# class TokenlearnModel(FinetunableStaticModel, pl.LightningModule):
#     """A TokenLearn model that learns to map token embeddings to target vectors."""

#     def __init__(
#         self,
#         *,
#         vectors: torch.Tensor,
#         tokenizer: Tokenizer,
#         out_dim: int = 2,
#         pad_id: int = 0,
#         lr_model: float = 3e-3,
#         lr_linear: float = 1e-2,
#         cosine_weight: float = 1.0,
#         mse_weight: float = 1.0,
#     ) -> None:
#         """
#         Initialize a TokenlearnModel.

#         :param vectors: (vocab_size, embed_dim) float tensor
#         :param tokenizer: the tokenizer
#         :param out_dim: dimension of the linear head output
#         :param pad_id: the padding ID
#         :param lr_model: learning rate for the embeddings + weighting param
#         :param lr_linear: learning rate for the linear head
#         :param cosine_weight: weight for the (1 - cosSim) loss
#         :param mse_weight: weight for the MSE penalty on the mean embedding
#         """
#         super().__init__(vectors=vectors, tokenizer=tokenizer, out_dim=out_dim, pad_id=pad_id)
#         self.lr_model = lr_model
#         self.lr_linear = lr_linear
#         self.cosine_weight = cosine_weight
#         self.mse_weight = mse_weight

#     def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
#         """The training step for the model."""
#         input_ids, target_vectors = batch
#         y_hat, mean_emb = self.forward(input_ids)  # calls FinetunableStaticModel.forward
#         # 1) Cosine similarity loss between y_hat and target_vectors
#         cos_loss = 1.0 - F.cosine_similarity(y_hat, target_vectors, dim=1).mean()
#         # 2) MSE penalty on embeddings
#         mse_loss = (mean_emb**2).mean()

#         loss = self.cosine_weight * cos_loss + self.mse_weight * mse_loss
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def configure_optimizers(self) -> torch.optim.Adam:
#         """Configure the optimizer for the model."""
#         model_params = list(self.embeddings.parameters()) + [self.w]
#         linear_params = self.head.parameters()

#         optimizer = torch.optim.Adam(
#             [
#                 {"params": model_params, "lr": self.lr_model},
#                 {"params": linear_params, "lr": self.lr_linear},
#             ]
#         )
#         return optimizer

#     @classmethod
#     def from_static_model(
#         cls: type[ModelType],
#         model: StaticModel,
#         out_dim: Optional[int] = None,
#         lr_model: float = 3e-3,
#         lr_linear: float = 1e-2,
#         cosine_weight: float = 1.0,
#         mse_weight: float = 1.0,
#     ) -> ModelType:
#         """Method to wrap a StaticModel in a TokenlearnModel."""
#         if out_dim is None:
#             out_dim = model.dim
#         embeddings = torch.from_numpy(np.nan_to_num(model.embedding))
#         return cls(
#             vectors=embeddings,
#             tokenizer=model.tokenizer,
#             out_dim=out_dim,
#             pad_id=model.tokenizer.token_to_id("[PAD]"),
#             lr_model=lr_model,
#             lr_linear=lr_linear,
#             cosine_weight=cosine_weight,
#             mse_weight=mse_weight,
#         )

# def apply_weighting(self, texts: list[str], alpha: float = 1e-3, pca_dims: int = 256) -> None:
#     """
#     Post-training step to apply SIF weighting + PCA to the embeddings.

#     Does the following:
#     1) Count token frequencies across `texts`.
#     2) PCA transform embeddings to `pca_dims`.
#     3) Weight each token by alpha / (alpha + p(token)).
#     4) Place the new embeddings back into self.embeddings.weight.

#     :param texts: The corpus on which to compute frequencies.
#     :param alpha: The SIF alpha.
#     :param pca_dims: The dimensionality to keep after PCA.
#     """
#     logger.info("Applying SIF weighting + PCA to the embeddings.")
#     # 1) freq distribution
#     probas = calculate_token_probabilities(self.tokenizer, texts)

#     # 2) fetch current embeddings as np array
#     current_emb = self.embeddings.weight.detach().cpu().numpy()
#     current_emb = np.nan_to_num(current_emb)

#     # 3) PCA to `pca_dims`
#     p = PCA(n_components=pca_dims)
#     emb_pca = p.fit_transform(current_emb)  # shape: [vocab_size, pca_dims]

#     # 4) SIF weighting
#     # alpha / (alpha + freq)
#     f = alpha / (alpha + probas)
#     emb_pca *= f[:, None]

#     # 5) store back into self.embeddings.weight
#     with torch.no_grad():
#         new_emb = torch.from_numpy(emb_pca).float()
#         if new_emb.shape[1] == self.embed_dim:
#             # If pca_dims == self.embed_dim, we can store back directly.
#             self.embeddings.weight.copy_(new_emb.to(self.embeddings.weight.device))
#         else:
#             # If pca_dims != embed_dim,
#             # we must either resize self.head, or store it but break the shape.
#             # Typically you'd want pca_dims == self.embed_dim for a direct overwrite.
#             logger.warning(
#                 f"PCA dims={pca_dims} != embed_dim={self.embed_dim}."
#                 " You have changed the embedding dimension. The linear head may need updating."
#             )
#             # For simplicity, let's just replace and assume user knows they changed dim.
#             self.embeddings = nn.Embedding.from_pretrained(new_emb, freeze=False, padding_idx=self.pad_id)
#             self.embed_dim = pca_dims
#             # Optionally also re-init self.head if you want out_dim <-> embed_dim consistent.

#     logger.info(f"SIF weighting + PCA complete. New embedding shape: {self.embeddings.weight.shape}.")

#     def fit(
#         self,
#         dataset: TextDataset,
#         batch_size: int = 32,
#         max_epochs: int = 5,
#         device: str = "cpu",
#         patience: Optional[int] = None,
#     ) -> None:
#         """Fit the model."""
#         callbacks = []
#         if patience is not None:
#             callbacks.append(EarlyStopping(monitor="train_loss", mode="min", patience=patience))

#         train_loader = dataset.to_dataloader(batch_size=batch_size, shuffle=True)

#         trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, callbacks=callbacks)
#         trainer.fit(self, train_loader)

#     def save_pretrained(self, save_directory: str) -> None:
#         """Convert the trained model to a StaticModel and save it."""
#         final_static = self.to_static_model()
#         final_static.save_pretrained(save_directory)
#         logger.info(f"Saved TokenlearnModel as a static model to '{save_directory}'")


# def run() -> None:
#     """Run the tokenlearn training script."""
#     # Initialize a StaticModel
#     model = StaticModel.from_pretrained("minishlab/M2V_base_output")

#     # Collect paths for training data
#     paths = sorted(Path("../tokenlearn/data/c4_features_bgebase_test").glob("*.json"))
#     X, y = collect_means_and_texts(paths)

#     # Convert to TokenlearnDataset
#     y = torch.tensor(y)
#     dataset = TokenlearnDataset(X, y, tokenizer=model.tokenizer)

#     # Wrap it in our TokenlearnModel
#     tokenlearn_model = TokenlearnModel.from_static_model(model, out_dim=y.shape[1])

#     # Fit the model
#     tokenlearn_model.fit(dataset, batch_size=8, max_epochs=1)

#     # Apply weighting (SIF + PCA)
#     tokenlearn_model.apply_weighting(X, alpha=1e-3, pca_dims=256)

#     # Save the final static model
#     tokenlearn_model.save_pretrained("local/tokenlearn_test")


# if __name__ == "__main__":
#     run()
