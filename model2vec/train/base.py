from __future__ import annotations

from typing import Any, TypeVar

import torch
from tokenizers import Encoding, Tokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model2vec import StaticModel


class FinetunableStaticModel(nn.Module):
    def __init__(self, *, vectors: torch.Tensor, tokenizer: Tokenizer, out_dim: int, pad_id: int = 0) -> None:
        """
        Initialize a trainable StaticModel from a StaticModel.

        :param vectors: The embeddings of the staticmodel.
        :param tokenizer: The tokenizer.
        :param out_dim: The output dimension of the head.
        :param pad_id: The padding id. This is set to 0 in almost all model2vec models
        """
        super().__init__()
        self.pad_id = pad_id
        self.out_dim = out_dim
        self.embed_dim = vectors.shape[1]

        self.embeddings = nn.Embedding.from_pretrained(vectors.clone(), freeze=False, padding_idx=pad_id)
        self.head = self.construct_head()

        # Weights for
        weights = torch.ones(len(vectors))
        weights[pad_id] = 0
        self.w = nn.Parameter(weights)
        self.tokenizer = tokenizer

    def construct_head(self) -> nn.Module:
        """Method should be overridden for various other classes."""
        return nn.Linear(self.embed_dim, self.out_dim)

    @classmethod
    def from_pretrained(
        cls: type[ModelType], out_dim: int, model_name: str = "minishlab/potion-base-8m", **kwargs: Any
    ) -> ModelType:
        """Load the model from a pretrained model2vec model."""
        model = StaticModel.from_pretrained(model_name)
        return cls.from_static_model(model, out_dim, **kwargs)

    @classmethod
    def from_static_model(cls: type[ModelType], model: StaticModel, out_dim: int, **kwargs: Any) -> ModelType:
        """Load the model from a static model."""
        embeddings_converted = torch.from_numpy(model.embedding)
        return cls(
            vectors=embeddings_converted,
            pad_id=model.tokenizer.token_to_id("[PAD]"),
            out_dim=out_dim,
            tokenizer=model.tokenizer,
            **kwargs,
        )

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        A forward pass and mean pooling.

        This function is analogous to `StaticModel.encode`, but reimplemented to allow gradients
        to pass through.

        :param input_ids: A 2D tensor of input ids. All input ids are have to be within bounds.
        :return: The mean over the input ids, weighted by token weights.
        """
        w = self.w[input_ids]
        zeros = (input_ids != self.pad_id).float()
        length = zeros.sum(1)
        embedded = self.embeddings(input_ids)
        # Simulate actual mean
        # Zero out the padding
        embedded = embedded * zeros[:, :, None]
        embedded = (embedded * w[:, :, None]).sum(1) / w.sum(1)[:, None]
        embedded = embedded / length[:, None]

        return torch.nn.functional.normalize(embedded)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean, and a classifier layer after."""
        encoded = self._encode(input_ids)
        return self.head(encoded), encoded

    def tokenize(self, texts: list[str], max_length: int | None = 512) -> torch.Tensor:
        """
        Tokenize a bunch of strings into a single padded 2D tensor.

        Note that this is not used during training.

        :param texts: The texts to tokenize.
        :param max_length: If this is None, the sequence lengths are truncated to 512.
        :return: A 2D padded tensor
        """
        encoded: list[Encoding] = self.tokenizer.encode_batch_fast(texts, add_special_tokens=False)
        encoded_ids: list[torch.Tensor] = [torch.Tensor(encoding.ids[:max_length]).long() for encoding in encoded]
        return pad_sequence(encoded_ids, batch_first=True)

    @property
    def device(self) -> str:
        """Get the device of the model."""
        return self.embeddings.weight.device

    def to_static_model(self, config: dict[str, Any] | None = None) -> StaticModel:
        """
        Convert the model to a static model.

        This is useful if you want to discard your head, and consolidate the information learned by
        the model to use it in a downstream task.

        :param config: The config used in the StaticModel. If this is set to None, it will have no config.
        :return: A static model.
        """
        # Perform the forward pass on the selected device.
        with torch.no_grad():
            all_indices = torch.arange(len(self.embeddings.weight))[:, None].to(self.device)
            vectors = self._encode(all_indices).cpu().numpy()

        new_model = StaticModel(vectors=vectors, tokenizer=self.tokenizer, config=config)

        return new_model


class TextDataset(Dataset):
    def __init__(self, texts: list[str], targets: torch.Tensor, tokenizer: Tokenizer) -> None:
        """
        A dataset of texts.

        This dataset tokenizes the texts and stores them as Tensors, which are then padded in the collation function.

        :param texts: The texts to tokenize.
        :param targets: The targets.
        :param tokenizer: The tokenizer to use.
        :raises ValueError: If the number of labels does not match the number of texts.
        """
        if len(targets) != len(texts):
            raise ValueError("Number of labels does not match number of texts.")
        self.texts = texts
        self.tokenized_texts: list[list[int]] = [
            encoding.ids for encoding in tokenizer.encode_batch_fast(self.texts, add_special_tokens=False)
        ]
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.tokenized_texts)

    def __getitem__(self, index: int) -> tuple[list[int], torch.Tensor]:
        """Gets an item."""
        return self.tokenized_texts[index], self.targets[index]

    @staticmethod
    def collate_fn(batch: list[tuple[list[list[int]], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        texts, targets = zip(*batch)

        tensors = [torch.LongTensor(x) for x in texts]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)

        return padded, torch.stack(targets)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        return DataLoader(self, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)


ModelType = TypeVar("ModelType", bound=FinetunableStaticModel)