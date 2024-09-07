# -*- coding: utf-8 -*-
from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from reach import Reach
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm
from wordfreq import word_frequency

from korok.model2vec.tokenizer import Model2VecTokenizer, create_model2vec_tokenizer_from_vocab
from korok.model2vec.utils import (
    add_token_to_reach,
    safe_load_reach,
)

PathLike = Path | str


logger = getLogger(__name__)


class StaticEmbedder:
    def __init__(self, vectors: Reach, tokenizer: Model2VecTokenizer) -> None:
        """
        Initialize the StaticEmbedder.

        :param vectors: The Reach vectors to use.
        :param tokenizer: The Transformers tokenizer to use.
        """
        self.vectors = vectors
        self.tokenizer = tokenizer
        self.unk_token = self.vectors.indices[self.vectors.unk_index]

    @property
    def name(self) -> str:
        """Return the name of the vectors."""
        return self.vectors.name

    @classmethod
    def from_vectors(
        cls: type[StaticEmbedder],
        vector_path: PathLike,
        apply_pca: bool = True,
        apply_zipf: bool = True,
        apply_frequency: bool = False,
    ) -> StaticEmbedder:
        """
        Create a static embeddder by creating a word-level tokenizer.

        :param vector_path: The path to the vectors.
        :param apply_pca: Whether to apply PCA to the vectors.
        :param apply_zipf: Whether to apply Zipf weighting to the vectors.
        :param apply_frequency: Whether to apply frequency weighting to the vectors.
        :return: A StaticEmbedder
        :raises ValueError: If both apply_zipf and apply_frequency are True.
        """
        path = Path(vector_path)
        embeddings = safe_load_reach(path)

        embeddings = add_token_to_reach(embeddings, "[PAD]", set_as_unk=False)
        embeddings = add_token_to_reach(embeddings, "[UNK]", set_as_unk=True)

        if apply_pca and embeddings.size > 300:
            p = PCA(n_components=300, whiten=False)
            embeddings._vectors = p.fit_transform(embeddings._vectors)

        if apply_zipf and apply_frequency:
            raise ValueError("Cannot apply both zipf and frequency weighting.")

        if apply_zipf:
            # NOTE: zipf weighting
            w = np.log(np.arange(1, len(embeddings) + 1))
            embeddings._vectors *= w[:, None]
        if apply_frequency:
            weight = np.zeros(len(embeddings))
            for idx, word in enumerate(embeddings.sorted_items):
                weight[idx] = word_frequency(word, "en")
            embeddings._vectors *= np.log(1 / weight[:, None])

        tokenizer = create_model2vec_tokenizer_from_vocab(embeddings.items, unk_token="[UNK]", pad_token="[PAD]")
        return cls(embeddings, tokenizer)

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """
        Encode a list of sentences.

        :param sentences: The list of sentences to encode.
        :param **kwargs: Additional keyword arguments.
        :return: The encoded sentences.
        """
        output = []
        for sentence in tqdm(sentences):
            tokens = [token for token in self.tokenizer.tokenize(sentence) if token != self.unk_token][:512]
            vector = self.vectors.mean_pool(tokens, safeguard=False)
            output.append(vector)

        return np.stack(output)


Encoder: TypeAlias = StaticEmbedder | SentenceTransformer


def load_embedder(model_path: str, word_level: bool, device: str = "cpu") -> tuple[Encoder, str]:
    """
    Load the embedder.

    :param model_path: The path to the model.
    :param word_level: Whether to use word level embeddings.
    :param device: The device to use.
    :return: The embedder and the name of the model.
    """
    embedder: Encoder

    if word_level:
        logger.info("Loading word level model")
        embedder = StaticEmbedder.from_vectors(model_path, apply_pca=True, apply_zipf=True)
        name = embedder.name
    else:
        logger.info("Loading SentenceTransformer model")
        # Always load on CPU
        embedder = SentenceTransformer(model_name_or_path=model_path, device="cpu")
        embedder = embedder.eval().to(device)
        model_name = Path(model_path).name.replace("_", "-")
        name = f"sentencetransformer_{model_name}"

    return embedder, name
