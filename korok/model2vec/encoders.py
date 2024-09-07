# -*- coding: utf-8 -*-
from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from wordfreq import word_frequency

from korok.model2vec.tokenizer import create_model2vec_tokenizer_from_vocab

PathLike = Path | str


logger = getLogger(__name__)


class StaticEmbedder:
    def __init__(self, tokens: list[str], vectors: np.ndarray, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        Initialize the StaticEmbedder.

        :param tokens: The tokens to use.
        :param vectors: The vectors to use.
        :param tokenizer: The Transformers tokenizer to use.
        """
        self.vectors = vectors
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.unk_token = self.tokenizer.unk_token
        self._index = {token: idx for idx, token in enumerate(tokens)}
        self.unk_index = self._index[self.unk_token]

    @classmethod
    def from_dataset(
        cls: type[StaticEmbedder],
        path: PathLike,
        apply_pca: bool = True,
        apply_zipf: bool = True,
        apply_frequency: bool = False,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
    ) -> StaticEmbedder:
        """
        Create a static embeddder by loading a dataset from disk or from the hub.

        :param path: The path to the dataset to create the tokenizer from.
        :param apply_pca: Whether to apply PCA to the vectors.
        :param apply_zipf: Whether to apply Zipf weighting to the vectors.
        :param apply_frequency: Whether to apply frequency weighting to the vectors.
        :param unk_token: The unk token to use.
        :param pad_token: The pad token to use.
        :return: A StaticEmbedder
        """
        path = Path(path)
        if path.is_dir():
            dataset = load_from_disk(path)
        else:
            dataset = load_dataset(path)

        tokens, vectors = dataset["tokens"], dataset["vectors"]
        tokens = list(tokens)
        vectors = np.stack(vectors)
        return cls.from_vectors(tokens, vectors, apply_pca, apply_zipf, apply_frequency, unk_token, pad_token)

    @classmethod
    def from_vectors(
        cls: type[StaticEmbedder],
        tokens: list[str],
        vectors: np.ndarray,
        apply_pca: bool = True,
        apply_zipf: bool = True,
        apply_frequency: bool = False,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
    ) -> StaticEmbedder:
        """
        Create a static embeddder by creating a word-level tokenizer.

        :param tokens: The tokens to use.
        :param vectors: The vectors to use.
        :param apply_pca: Whether to apply PCA to the vectors.
        :param apply_zipf: Whether to apply Zipf weighting to the vectors.
        :param apply_frequency: Whether to apply frequency weighting to the vectors.
        :param unk_token: The unk token to use.
        :param pad_token: The pad token to use.
        :return: A StaticEmbedder
        :raises ValueError: If both apply_zipf and apply_frequency are True.
        """
        if unk_token not in tokens:
            tokens.append(unk_token)
            vectors = np.vstack([vectors, np.zeros(vectors.shape[1])])
        if pad_token not in tokens:
            tokens.append(pad_token)
            vectors = np.vstack([vectors, np.zeros(vectors.shape[1])])

        if apply_pca and vectors.shape[1] > 300:
            p = PCA(n_components=300, whiten=False)
            vectors = p.fit_transform(vectors)

        if apply_zipf and apply_frequency:
            raise ValueError("Cannot apply both zipf and frequency weighting.")

        if apply_zipf:
            # NOTE: zipf weighting
            w = np.log(np.arange(1, len(vectors) + 1))
            vectors *= w[:, None]
        if apply_frequency:
            weight = np.zeros(len(vectors))
            for idx, word in enumerate(tokens):
                weight[idx] = word_frequency(word, "en")
            vectors *= np.log(1 / weight[:, None])

        tokenizer = create_model2vec_tokenizer_from_vocab(tokens, unk_token=unk_token, pad_token=pad_token)

        return cls(tokens, vectors, tokenizer)

    def dim(self) -> int:
        """
        Get the dimension of the vectors.

        :return: The dimension of the vectors.
        """
        return self.vectors.shape[1]

    def encode(self, sentences: list[str], show_progressbar: bool = True, **kwargs: Any) -> np.ndarray:
        """
        Encode a list of sentences.

        :param sentences: The list of sentences to encode.
        :param show_progressbar: Whether to show the progress bar.
        :param **kwargs: Additional keyword arguments.
        :return: The encoded sentences.
        """
        output = []
        for sentence in tqdm(sentences, disable=not show_progressbar):
            encoded = self.tokenizer.encode(sentence, add_special_tokens=False)
            encoded = [index for index in encoded if index != self.unk_index]
            if not encoded:
                logger.warning(f"Got empty tokens for sentence {sentence}")
                vector = np.zeros_like(self.dim)
            else:
                vector = np.mean(self.vectors[encoded], axis=0)
            output.append(vector)

        return np.stack(output)
