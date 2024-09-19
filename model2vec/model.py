from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tokenizers import Encoding, Tokenizer
from torch.nn import EmbeddingBag
from tqdm import tqdm

from model2vec.utils import load_pretrained, save_pretrained

PathLike = Path | str


logger = getLogger(__name__)


class StaticModel:
    def __init__(self, vectors: np.ndarray, tokenizer: Tokenizer, config: dict[str, Any]) -> None:
        """
        Initialize the StaticModel.

        :param vectors: The vectors to use.
        :param tokenizer: The Transformers tokenizer to use.
        :param config: Any metadata config.
        :raises: ValueError if the number of tokens does not match the number of vectors.
        """
        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        self.vectors = vectors
        self.tokens = tokens
        self.embedding = EmbeddingBag.from_pretrained(torch.from_numpy(vectors))

        if len(tokens) != vectors.shape[0]:
            raise ValueError(f"Number of tokens ({len(tokens)}) does not match number of vectors ({vectors.shape[0]})")

        self.tokenizer = tokenizer
        self.max_token_length = max(len(token) for token in tokens)
        self.unk_token_id: int | None
        if getattr(self.tokenizer.model, "unk_token") and self.tokenizer.model.unk_token is not None:
            self.unk_token_id = tokenizer.get_vocab()[self.tokenizer.model.unk_token]
        else:
            self.unk_token_id = None

        self.config = config

    def save_pretrained(self, path: PathLike) -> None:
        """
        Save the pretrained model.

        :param path: The path to save to.
        """
        save_pretrained(Path(path), self.vectors, self.tokenizer, self.config)

    @classmethod
    def from_pretrained(
        cls: type[StaticModel],
        path: PathLike,
        huggingface_token: str | None = None,
    ) -> StaticModel:
        """
        Create a static embeddder by creating a word-level tokenizer.

        :param path: The path to load your static model from.
        :param huggingface_token: The huggingface token to use.
        :return: A StaticEmbedder
        """
        embeddings, tokenizer, config = load_pretrained(path, huggingface_token=huggingface_token)

        return cls(embeddings, tokenizer, config)

    @property
    def dim(self) -> int:
        """
        Get the dimension of the vectors.

        :return: The dimension of the vectors.
        """
        return self.vectors.shape[1]

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        """Normalize an array to unit length."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1

        return X / norms

    def encode(
        self,
        sentences: list[str] | str,
        show_progressbar: bool = False,
        max_length: int | None = 512,
        norm: bool = False,
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
        :param norm: Whether to normalize the embeddings to unit length.
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

        if norm:
            out_array = self.normalize(out_array)

        if was_single:
            return out_array[0]

        return out_array

    def _encode_batch(self, sentences: list[str], max_length: int | None) -> np.ndarray:
        """Encode a batch of sentences."""
        encodings: list[Encoding] = self.tokenizer.encode_batch(sentences, add_special_tokens=False)
        encodings_ids = [encoding.ids for encoding in encodings]

        if self.unk_token_id is not None:
            # NOTE: Remove the unknown token: necessary for word-level models.
            encodings_ids = [
                [token_id for token_id in token_ids if token_id != self.unk_token_id] for token_ids in encodings_ids
            ]
        if max_length is not None:
            encodings_ids = [token_ids[:max_length] for token_ids in encodings_ids]

        offsets = np.cumsum([0] + [len(token_ids) for token_ids in encodings_ids[:-1]])
        ids = torch.tensor([token_id for token_ids in encodings_ids for token_id in token_ids], dtype=torch.long)

        return self.embedding(ids, torch.tensor(offsets, dtype=torch.long)).detach().numpy()

    @staticmethod
    def _batch(sentences: list[str], batch_size: int) -> Iterator[list[str]]:
        """Batch the sentences into equal-sized."""
        return (sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size))
