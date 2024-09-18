from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from model2vec.utils import load_pretrained, save_pretrained

PathLike = Path | str


logger = getLogger(__name__)


class StaticModel:
    def __init__(self, vectors: np.ndarray, tokenizer: PreTrainedTokenizerFast, config: dict[str, Any]) -> None:
        """
        Initialize the StaticEmbedder.

        :param vectors: The vectors to use.
        :param tokenizer: The Transformers tokenizer to use.
        :param config: Any metadata config.
        :raises: ValueError if the number of tokens does not match the number of vectors.
        """
        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        self.vectors = vectors
        self.tokens = tokens

        if len(tokens) != vectors.shape[0]:
            raise ValueError(f"Number of tokens ({len(tokens)}) does not match number of vectors ({vectors.shape[0]})")

        self.tokenizer = tokenizer
        self.unk_token_id = tokenizer.get_vocab()[self.tokenizer.unk_token]
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
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of sentences.

        :param sentences: The list of sentences to encode.
        :param show_progressbar: Whether to show the progress bar.
        :param max_length: The maximum length of the sentences. Any tokens beyond this length will be truncated.
        :param norm: Whether to normalize the embeddings to unit length.
        :param **kwargs: Any additional arguments. These are ignored.
        :return: The encoded sentences.
        """
        was_single = False
        if isinstance(sentences, str):
            sentences = [sentences]
            was_single = True

        out_array = np.zeros((len(sentences), self.dim))
        for idx, sentence in enumerate(tqdm(sentences, disable=not show_progressbar)):
            encoded = self.tokenizer.encode(
                sentence, add_special_tokens=False, max_length=max_length, truncation=bool(max_length)
            )
            # NOTE: Remove the unknown token: necessary for word-level models.
            encoded = [token_id for token_id in encoded if token_id != self.unk_token_id]
            if not encoded:
                logger.info(f"Got empty tokens for sentence: `{sentence}`")
                continue
            out_array[idx] = np.mean(self.vectors[encoded], axis=0)

        if norm:
            out_array = self.normalize(out_array)

        if was_single:
            return out_array[0]

        return out_array
