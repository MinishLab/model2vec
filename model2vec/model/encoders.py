# -*- coding: utf-8 -*-
from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
from reach import Reach
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer
from wordfreq import word_frequency

from model2vec.model.tokenizer import Model2VecTokenizer, create_model2vec_tokenizer
from model2vec.model.utilities import (
    add_token_to_reach,
    create_input_embeddings_from_model_name,
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
        This function creates a static embeddder by creating a word-level tokenizer.

        :param vectors: A reach vector instance.
        :return: A StaticEmbedder
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
            embeddings._vectors *= np.log(np.arange(1, len(embeddings) + 1))[:, None]
        if apply_frequency:
            weight = np.zeros(len(embeddings))
            for idx, word in enumerate(embeddings.sorted_items):
                weight[idx] = word_frequency(word, "en")
            embeddings._vectors *= np.log(1 / weight[:, None])

        tokenizer = create_model2vec_tokenizer(embeddings.items, unk_token="[UNK]", pad_token="[PAD]")
        return cls(embeddings, tokenizer)

    @classmethod
    def from_model(
        cls: type[StaticEmbedder], model_name: PathLike, module_path: tuple[str, ...] | None = None
    ) -> StaticEmbedder:
        """
        Classmethod to create a StaticEmbedder from a model name.

        :param model_name: The model name to use.
        :param module_path: The module path to use.
        :return: A StaticEmbedder.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if module_path is not None:
            embeddings = create_input_embeddings_from_model_name(model_name, module_path)
        else:
            embeddings = create_input_embeddings_from_model_name(model_name)

        return cls(embeddings, tokenizer)

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """
        Encode a list of sentences.

        :param sentences: The list of sentences to encode.
        :return: The encoded sentences.
        """
        output = []
        for sentence in tqdm(sentences):
            tokens = [token for token in self.tokenizer.tokenize(sentence) if token != self.unk_token][:512]
            vector = self.vectors.mean_pool(tokens, safeguard=False)
            output.append(vector)

        return np.stack(output)
