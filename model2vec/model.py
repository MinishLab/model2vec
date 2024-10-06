from __future__ import annotations

from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np
import torch
from tokenizers import Encoding, Tokenizer
from torch import nn
from torch.nn import Embedding, EmbeddingBag
from tqdm import tqdm

from model2vec.utils import load_pretrained, push_folder_to_hub, save_pretrained

PathLike = Path | str


logger = getLogger(__name__)


class StaticModel(nn.Module):
    def __init__(
        self,
        vectors: np.ndarray,
        tokenizer: Tokenizer,
        config: dict[str, Any] | None = None,
        normalize: bool | None = None,
        base_model_name: str | None = None,
        language: list[str] | None = None,
    ) -> None:
        """
        Initialize the StaticModel.

        :param vectors: The vectors to use.
        :param tokenizer: The Transformers tokenizer to use.
        :param config: Any metadata config.
        :param normalize: Whether to normalize.
        :param base_model_name: The used base model name. Used for creating a model card.
        :param language: The language of the model. Used for creating a model card.
        :raises: ValueError if the number of tokens does not match the number of vectors.
        """
        super().__init__()
        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        self.tokens = tokens
        tensor = torch.from_numpy(vectors)

        # NOTE: We make two embedding modules, but since they share the same memory,
        # there's no overhead.
        # Gradients are also integrated into both.
        self.embedding_bag = EmbeddingBag.from_pretrained(tensor, mode="mean")
        self.embedding = Embedding.from_pretrained(tensor)

        if len(tokens) != vectors.shape[0]:
            raise ValueError(f"Number of tokens ({len(tokens)}) does not match number of vectors ({vectors.shape[0]})")

        self.tokenizer = tokenizer
        self.unk_token_id: int | None
        if hasattr(self.tokenizer.model, "unk_token") and self.tokenizer.model.unk_token is not None:
            self.unk_token_id = tokenizer.get_vocab()[self.tokenizer.model.unk_token]
        else:
            self.unk_token_id = None

        self.median_token_length = int(np.median([len(token) for token in self.tokens]))
        self.config = config or {}
        self.base_model_name = base_model_name
        self.language = language

        if normalize is not None:
            self.normalize = normalize
        else:
            self.normalize = self.config.get("normalize", False)

    @property
    def dim(self) -> int:
        """Get the dimension of the model."""
        return self.embedding.weight.shape[1]

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

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

    def save_pretrained(self, path: PathLike, model_name: str | None = None) -> None:
        """
        Save the pretrained model.

        :param path: The path to save to.
        :param model_name: The model name to use in the Model Card.
        """
        save_pretrained(
            folder_path=Path(path),
            embeddings=self.embedding.weight.numpy(),
            tokenizer=self.tokenizer,
            config=self.config,
            base_model_name=self.base_model_name,
            language=self.language,
            model_name=model_name,
        )

    def forward(self, X: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Helper method to facilitate training.

        :param X: a tuple of ids and offsets.
        :return: A padded output tensor.
        """
        ids, offsets = X
        return self.embedding_bag(ids, offsets)

    def forward_mean(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param ids: The input tensor.
        :param offsets: The offsets tensor.
        :return: The output tensor.
        """
        means = self.embedding_bag(ids, offsets)

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
        if max_length is not None:
            m = max_length * self.median_token_length
            sentences = [sentence[:m] for sentence in sentences]

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
        embeddings, tokenizer, config, metadata = load_pretrained(path, token=token)

        return cls(
            embeddings, tokenizer, config, base_model_name=metadata.get("base_model"), language=metadata.get("language")
        )

    def encode_as_sequence(
        self, sentences: list[str] | str, max_length: int | None = None
    ) -> list[np.ndarray] | np.ndarray:
        """
        Encode a list of sentences as a list of numpy arrays of tokens.

        This is useful if you want to use the tokens for further processing, or if you want to do sequence
        modeling.
        Note that if you just want the mean, you should use the `encode` method.
        This is about twice as slow.
        Sentences that do not contain any tokens will be turned into an empty array.

        :param sentences: The list of sentences to encode.
        :param max_length: The maximum length of the sentences. Any tokens beyond this length will be truncated.
            If this is None, no truncation is done.
        :return: The encoded sentences with an embedding per token.
        """
        was_single = False
        if isinstance(sentences, str):
            was_single = True
            sentences = [sentences]

        ids, offsets = self.tokenize(sentences=sentences, max_length=max_length)
        ids = ids.to(self.device)
        offsets = offsets.to(self.device)

        out = [tensor.cpu().numpy() for tensor in self._sub_encode_as_sequence(ids, offsets)]

        if was_single:
            return out[0]

        return out

    def _sub_encode_as_sequence(self, ids: torch.Tensor, offsets: torch.Tensor) -> list[torch.Tensor]:
        """Helper function to reduce deduplication."""
        out = []
        for x in range(len(offsets) - 1):
            start, end = offsets[x], offsets[x + 1]
            out.append(self.embedding(ids[start:end]))
        out.append(self.embedding(ids[offsets[-1] :]))

        return out

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
        ids = ids.to(self.device)
        offsets = offsets.to(self.device)
        return self.forward_mean(ids, offsets).cpu().numpy()

    @staticmethod
    def _batch(sentences: list[str], batch_size: int) -> Iterator[list[str]]:
        """Batch the sentences into equal-sized."""
        return (sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size))

    def push_to_hub(self, repo_id: str, private: bool = False, token: str | None = None) -> None:
        """
        Push the model to the huggingface hub.

        NOTE: you need to pass a token if you are pushing a private model.

        :param repo_id: The repo id to push to.
        :param private: Whether the repo, if created is set to private.
            If the repo already exists, this doesn't change the visibility.
        :param token: The huggingface token to use.
        """
        with TemporaryDirectory() as temp_dir:
            self.save_pretrained(temp_dir, model_name=repo_id)
            push_folder_to_hub(Path(temp_dir), repo_id, private, token)
