from __future__ import annotations

from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np
import torch
from tokenizers import Encoding, Tokenizer
from torch import nn
from torch.nn import EmbeddingBag
from tqdm import tqdm

from model2vec.utils import load_pretrained, push_folder_to_hub, save_pretrained

PathLike = Path | str


logger = getLogger(__name__)


class StaticModel(nn.Module):
    def __init__(
        self,
        vectors: np.ndarray,
        tokenizer: Tokenizer,
        config: dict[str, Any],
        normalize: bool | None = None,
        base_model_name: str | None = None,
    ) -> None:
        """
        Initialize the StaticModel.

        :param vectors: The vectors to use.
        :param tokenizer: The Transformers tokenizer to use.
        :param config: Any metadata config.
        :param normalize: Whether to normalize.
        :param base_model_name: The used base model name. Used for creating a model card.
        :raises: ValueError if the number of tokens does not match the number of vectors.
        """
        super().__init__()
        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        self.tokens = tokens
        self.embedding = EmbeddingBag.from_pretrained(torch.from_numpy(vectors))

        if len(tokens) != vectors.shape[0]:
            raise ValueError(f"Number of tokens ({len(tokens)}) does not match number of vectors ({vectors.shape[0]})")

        self.tokenizer = tokenizer
        self.unk_token_id: int | None
        if hasattr(self.tokenizer.model, "unk_token") and self.tokenizer.model.unk_token is not None:
            self.unk_token_id = tokenizer.get_vocab()[self.tokenizer.model.unk_token]
        else:
            self.unk_token_id = None

        self.config = config
        self.base_model_name = base_model_name

        if normalize is not None:
            self.normalize = normalize
        else:
            self.normalize = config.get("normalize", False)

    @property
    def normalize(self) -> bool:
        """
        Get the normalize value.

        :return: The normalize value.
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        """Update the config if the value of normalize changes."""
        config_normalize = self.config.get("normalize", False)
        self._normalize = value
        if config_normalize is not None and value != config_normalize:
            logger.warning(
                f"Set normalization to `{value}`, which does not match config value `{config_normalize}`. Updating config."
            )
        self.config["normalize"] = value

    def save_pretrained(self, path: PathLike) -> None:
        """
        Save the pretrained model.

        :param path: The path to save to.
        """
        save_pretrained(
            folder_path=Path(path),
            embeddings=self.embedding.weight.numpy(),
            tokenizer=self.tokenizer,
            config=self.config,
            base_model_name=self.base_model_name,
        )

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param ids: The input tensor.
        :param offsets: The offsets tensor.
        :return: The output tensor.
        """
        means = self.embedding(ids, offsets)
        if self.normalize:
            return torch.nn.functional.normalize(means)
        return means

    def tokenize(self, sentences: list[str], max_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a sentence.

        :param sentences: The sentence to tokenize.
        :param max_length: The maximum length of the sentence.
        :return: The tokens.
        """
        encodings: list[Encoding] = self.tokenizer.encode_batch(sentences, add_special_tokens=False)
        encodings_ids = [encoding.ids for encoding in encodings]

        if self.unk_token_id is not None:
            # NOTE: Remove the unknown token: necessary for word-level models.
            encodings_ids = [
                [token_id for token_id in token_ids if token_id != self.unk_token_id] for token_ids in encodings_ids
            ]
        if max_length is not None:
            encodings_ids = [token_ids[:max_length] for token_ids in encodings_ids]

        offsets = torch.from_numpy(np.cumsum([0] + [len(token_ids) for token_ids in encodings_ids[:-1]]))
        ids = torch.tensor([token_id for token_ids in encodings_ids for token_id in token_ids], dtype=torch.long)

        return ids, offsets

    @classmethod
    def from_pretrained(
        cls: type[StaticModel],
        path: PathLike,
        token: str | None = None,
    ) -> StaticModel:
        """
        Load a StaticModel from a local path or huggingface hub path.

        NOTE: if you load a private model from the huggingface hub, you need to pass a token.

        :param path: The path to load your static model from.
        :param token: The huggingface token to use.
        :return: A StaticEmbedder
        """
        embeddings, tokenizer, config = load_pretrained(path, token=token)

        return cls(embeddings, tokenizer, config)

    def encode(
        self,
        sentences: list[str] | str,
        show_progressbar: bool = False,
        max_length: int | None = 512,
        batch_size: int = 1024,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of sentences.

        This function encodes a list of sentences by averaging the word embeddings of the tokens in the sentence.
        For ease of use, we don't batch sentences together.

        :param sentences: The list of sentences to encode. You can also pass a single sentence.
        :param show_progressbar: Whether to show the progress bar.
        :param max_length: The maximum length of the sentences. Any tokens beyond this length will be truncated.
            If this is None, no truncation is done.
        :param batch_size: The batch size to use.
        :param **kwargs: Any additional arguments. These are ignored.
        :return: The encoded sentences. If a single sentence was passed, a vector is returned.
        """
        was_single = False
        if isinstance(sentences, str):
            sentences = [sentences]
            was_single = True

        out_arrays: list[np.ndarray] = []
        for batch in tqdm(
            self._batch(sentences, batch_size), total=(len(sentences) // batch_size) + 1, disable=not show_progressbar
        ):
            out_arrays.append(self._encode_batch(batch, max_length))

        out_array = np.concatenate(out_arrays, axis=0)

        if was_single:
            return out_array[0]

        return out_array

    @torch.no_grad()
    def _encode_batch(self, sentences: list[str], max_length: int | None) -> np.ndarray:
        """Encode a batch of sentences."""
        ids, offsets = self.tokenize(sentences, max_length)
        return self.forward(ids, offsets).numpy()

    @staticmethod
    def _batch(sentences: list[str], batch_size: int) -> Iterator[list[str]]:
        """Batch the sentences into equal-sized."""
        return (sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size))

    def push_to_hub(self, repo_id: str, token: str | None) -> None:
        """
        Push the model to the huggingface hub.

        NOTE: you need to pass a token if you are pushing a private model.

        :param repo_id: The repo id to push to.
        :param token: The huggingface token to use.
        """
        with TemporaryDirectory() as temp_dir:
            self.save_pretrained(temp_dir)
            push_folder_to_hub(Path(temp_dir), repo_id, token)
