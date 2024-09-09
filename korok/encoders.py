# -*- coding: utf-8 -*-
from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from wordfreq import word_frequency

from korok.utils import load_pretrained

PathLike = Path | str


logger = getLogger(__name__)


class StaticEmbedder:
    def __init__(self, vectors: np.ndarray, tokenizer: PreTrainedTokenizerFast, config: dict[str, Any]) -> None:
        """
        Initialize the StaticEmbedder.

        :param vectors: The vectors to use.
        :param tokenizer: The Transformers tokenizer to use.
        :param config: Any metadata config.
        """
        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        self.vectors = vectors
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.unk_index = tokenizer.get_vocab()[self.tokenizer.unk_token]
        self.config = config

    @classmethod
    def from_pretrained(
        cls: type[StaticEmbedder],
        path: PathLike,
        apply_pca: bool = True,
        apply_zipf: bool = True,
        apply_frequency: bool = False,
    ) -> StaticEmbedder:
        """
        Create a static embeddder by creating a word-level tokenizer.

        :param path: The path to point to.
        :param apply_pca: Whether to apply PCA to the vectors.
        :param apply_zipf: Whether to apply Zipf weighting to the vectors.
        :param apply_frequency: Whether to apply frequency weighting to the vectors.
        :return: A StaticEmbedder
        :raises ValueError: If both apply_zipf and apply_frequency are True.
        """
        path = Path(path)

        embeddings, tokenizer, config = load_pretrained(path)

        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))

        if apply_pca and embeddings.shape[1] > 300:
            p = PCA(n_components=300, whiten=False)
            embeddings = p.fit_transform(embeddings)

        if apply_zipf and apply_frequency:
            raise ValueError("Cannot apply both zipf and frequency weighting.")

        if apply_zipf:
            # NOTE: zipf weighting
            w = np.log(np.arange(1, len(embeddings) + 1))
            embeddings *= w[:, None]
        if apply_frequency:
            weight = np.zeros(len(embeddings))
            for idx, word in enumerate(tokens):
                weight[idx] = word_frequency(word, "en", minimum=1e-8)
            embeddings *= 1 / weight[:, None]

        return cls(embeddings, tokenizer, config)

    @property
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
            encoded = [index for index in encoded if index != self.unk_index][:512]
            if not encoded:
                logger.warning(f"Got empty tokens for sentence {sentence}")
                vector = np.zeros(self.dim)
            else:
                vector = np.mean(self.vectors[encoded], axis=0)
            output.append(vector)

        return np.stack(output)
