from __future__ import annotations

import math
import os
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Sequence, Union, overload

import numpy as np
from joblib import delayed
from tokenizers import Encoding, Tokenizer
from tqdm import tqdm

from model2vec.quantization import DType, quantize_and_reduce_dim
from model2vec.utils import ProgressParallel, load_local_model

PathLike = Union[Path, str]

logger = getLogger(__name__)


class StaticModel:
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
        :param normalize: Whether to normalize the embeddings.
        :param base_model_name: The used base model name. Used for creating a model card.
        :param language: The language of the model. Used for creating a model card.
        :raises: ValueError if the number of tokens does not match the number of vectors.
        """
        super().__init__()
        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        self.tokens = tokens

        self.embedding = vectors

        if len(tokens) != vectors.shape[0]:
            raise ValueError(f"Number of tokens ({len(tokens)}) does not match number of vectors ({vectors.shape[0]})")

        self.tokenizer = tokenizer
        self.unk_token_id: int | None
        if hasattr(self.tokenizer.model, "unk_token") and self.tokenizer.model.unk_token is not None:
            self.unk_token_id = tokenizer.get_vocab()[self.tokenizer.model.unk_token]
        else:
            self.unk_token_id = None  # pragma: no cover  # Doesn't actually happen, but can happen.

        self.median_token_length = int(np.median([len(token) for token in self.tokens]))
        self.config = config or {}
        self.base_model_name = base_model_name
        self.language = language
        if hasattr(self.tokenizer, "encode_batch_fast"):
            self._can_encode_fast = True
        else:
            self._can_encode_fast = False

        if normalize is not None:
            self.normalize = normalize
        else:
            self.normalize = self.config.get("normalize", False)

    @property
    def dim(self) -> int:
        """Get the dimension of the model."""
        return self.embedding.shape[1]

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
        config_normalize = self.config.get("normalize")
        self._normalize = value
        if config_normalize is not None and value != config_normalize:
            logger.warning(
                f"Set normalization to `{value}`, which does not match config value `{config_normalize}`. Updating config."
            )
        self.config["normalize"] = value

    def save_pretrained(self, path: PathLike, model_name: str | None = None, subfolder: str | None = None) -> None:
        """
        Save the pretrained model.

        :param path: The path to save to.
        :param model_name: The model name to use in the Model Card.
        :param subfolder: The subfolder to save to.
        """
        from model2vec.hf_utils import save_pretrained

        save_pretrained(
            folder_path=Path(path),
            embeddings=self.embedding,
            tokenizer=self.tokenizer,
            config=self.config,
            base_model_name=self.base_model_name,
            language=self.language,
            model_name=model_name,
            subfolder=subfolder,
        )

    def tokenize(self, sentences: Sequence[str], max_length: int | None = None) -> list[list[int]]:
        """
        Tokenize a list of sentences.

        :param sentences: The sentences to tokenize.
        :param max_length: The maximum length of the sentences in tokens. If this is None, sequences
            are not truncated.
        :return: A list of list of tokens.
        """
        if max_length is not None:
            m = max_length * self.median_token_length
            sentences = [sentence[:m] for sentence in sentences]

        if self._can_encode_fast:
            encodings: list[Encoding] = self.tokenizer.encode_batch_fast(sentences, add_special_tokens=False)
        else:
            encodings = self.tokenizer.encode_batch(sentences, add_special_tokens=False)

        encodings_ids = [encoding.ids for encoding in encodings]

        if self.unk_token_id is not None:
            # NOTE: Remove the unknown token: necessary for word-level models.
            encodings_ids = [
                [token_id for token_id in token_ids if token_id != self.unk_token_id] for token_ids in encodings_ids
            ]
        if max_length is not None:
            encodings_ids = [token_ids[:max_length] for token_ids in encodings_ids]

        return encodings_ids

    @classmethod
    def from_pretrained(
        cls: type[StaticModel],
        path: PathLike,
        token: str | None = None,
        normalize: bool | None = None,
        subfolder: str | None = None,
        quantize_to: str | DType | None = None,
        dimensionality: int | None = None,
    ) -> StaticModel:
        """
        Load a StaticModel from a local path or huggingface hub path.

        NOTE: if you load a private model from the huggingface hub, you need to pass a token.

        :param path: The path to load your static model from.
        :param token: The huggingface token to use.
        :param normalize: Whether to normalize the embeddings.
        :param subfolder: The subfolder to load from.
        :param quantize_to: The dtype to quantize the model to. If None, no quantization is done.
            If a string is passed, it is converted to a DType.
        :param dimensionality: The dimensionality of the model. If this is None, use the dimensionality of the model.
            This is useful if you want to load a model with a lower dimensionality.
            Note that this only applies if you have trained your model using mrl or PCA.
        :return: A StaticModel.
        """
        from model2vec.hf_utils import load_pretrained

        embeddings, tokenizer, config, metadata = load_pretrained(
            folder_or_repo_path=path,
            token=token,
            from_sentence_transformers=False,
            subfolder=subfolder,
        )

        embeddings = quantize_and_reduce_dim(
            embeddings=embeddings,
            quantize_to=quantize_to,
            dimensionality=dimensionality,
        )

        return cls(
            embeddings,
            tokenizer,
            config,
            normalize=normalize,
            base_model_name=metadata.get("base_model"),
            language=metadata.get("language"),
        )

    @classmethod
    def from_sentence_transformers(
        cls: type[StaticModel],
        path: PathLike,
        token: str | None = None,
        normalize: bool | None = None,
        quantize_to: str | DType | None = None,
        dimensionality: int | None = None,
    ) -> StaticModel:
        """
        Load a StaticModel trained with sentence transformers from a local path or huggingface hub path.

        NOTE: if you load a private model from the huggingface hub, you need to pass a token.

        :param path: The path to load your static model from.
        :param token: The huggingface token to use.
        :param normalize: Whether to normalize the embeddings.
        :param quantize_to: The dtype to quantize the model to. If None, no quantization is done.
            If a string is passed, it is converted to a DType.
        :param dimensionality: The dimensionality of the model. If this is None, use the dimensionality of the model.
            This is useful if you want to load a model with a lower dimensionality.
            Note that this only applies if you have trained your model using mrl or PCA.
        :return: A StaticModel.
        """
        from model2vec.hf_utils import load_pretrained

        embeddings, tokenizer, config, metadata = load_pretrained(
            folder_or_repo_path=path,
            token=token,
            from_sentence_transformers=True,
            subfolder=None,
        )

        embeddings = quantize_and_reduce_dim(
            embeddings=embeddings,
            quantize_to=quantize_to,
            dimensionality=dimensionality,
        )

        return cls(
            embeddings,
            tokenizer,
            config,
            normalize=normalize,
            base_model_name=metadata.get("base_model"),
            language=metadata.get("language"),
        )

    @overload
    def encode_as_sequence(
        self,
        sentences: str,
        max_length: int | None = None,
        batch_size: int = 1024,
        show_progress_bar: bool = False,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
    ) -> np.ndarray: ...

    @overload
    def encode_as_sequence(
        self,
        sentences: list[str],
        max_length: int | None = None,
        batch_size: int = 1024,
        show_progress_bar: bool = False,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
    ) -> list[np.ndarray]: ...

    def encode_as_sequence(
        self,
        sentences: str | list[str],
        max_length: int | None = None,
        batch_size: int = 1024,
        show_progress_bar: bool = False,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
    ) -> list[np.ndarray] | np.ndarray:
        """
        Encode a list of sentences as a list of numpy arrays of tokens.

        This is useful if you want to use the tokens for further processing, or if you want to do sequence
        modeling.
        Note that if you just want the mean, you should use the `encode` method.
        This is about twice as slow.
        Sentences that do not contain any tokens will be turned into an empty array.

        NOTE: the input type is currently underspecified. The actual input type is `Sequence[str] | str`, but this
            is not possible to implement in python typing currently.

        :param sentences: The list of sentences to encode.
        :param max_length: The maximum length of the sentences. Any tokens beyond this length will be truncated.
            If this is None, no truncation is done.
        :param batch_size: The batch size to use.
        :param show_progress_bar: Whether to show the progress bar.
        :param use_multiprocessing: Whether to use multiprocessing.
            By default, this is enabled for inputs > multiprocessing_threshold sentences and disabled otherwise.
        :param multiprocessing_threshold: The threshold in number of sentences for using multiprocessing.
        :return: The encoded sentences with an embedding per token.
        """
        was_single = False
        if isinstance(sentences, str):
            sentences = [sentences]
            was_single = True

        # Prepare all batches
        sentence_batches = list(self._batch(sentences, batch_size))
        total_batches = math.ceil(len(sentences) / batch_size)

        # Use joblib for multiprocessing if requested, and if we have enough sentences
        if use_multiprocessing and len(sentences) > multiprocessing_threshold:
            # Disable parallelism for tokenizers
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            results = ProgressParallel(n_jobs=-1, use_tqdm=show_progress_bar, total=total_batches)(
                delayed(self._encode_batch_as_sequence)(batch, max_length) for batch in sentence_batches
            )
            out_array: list[np.ndarray] = []
            for r in results:
                out_array.extend(r)
        else:
            out_array = []
            for batch in tqdm(
                sentence_batches,
                total=total_batches,
                disable=not show_progress_bar,
            ):
                out_array.extend(self._encode_batch_as_sequence(batch, max_length))

        if was_single:
            return out_array[0]
        return out_array

    def _encode_batch_as_sequence(self, sentences: Sequence[str], max_length: int | None) -> list[np.ndarray]:
        """Encode a batch of sentences as a sequence."""
        ids = self.tokenize(sentences=sentences, max_length=max_length)
        out: list[np.ndarray] = []
        for id_list in ids:
            if id_list:
                out.append(self.embedding[id_list])
            else:
                out.append(np.zeros((0, self.dim)))

        return out

    def encode(
        self,
        sentences: Sequence[str],
        show_progress_bar: bool = False,
        max_length: int | None = 512,
        batch_size: int = 1024,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of sentences.

        This function encodes a list of sentences by averaging the word embeddings of the tokens in the sentence.
        For ease of use, we don't batch sentences together.

        NOTE: the return type is currently underspecified. In the case of a single string, this returns a 1D array,
            but in the case of a list of strings, this returns a 2D array. Not possible to implement in numpy currently.

        :param sentences: The list of sentences to encode. You can also pass a single sentence.
        :param show_progress_bar: Whether to show the progress bar.
        :param max_length: The maximum length of the sentences. Any tokens beyond this length will be truncated.
            If this is None, no truncation is done.
        :param batch_size: The batch size to use.
        :param use_multiprocessing: Whether to use multiprocessing.
            By default, this is enabled for inputs > multiprocessing_threshold sentences and disabled otherwise.
        :param multiprocessing_threshold: The threshold in number of sentences for using multiprocessing.
        :param **kwargs: Any additional arguments. These are ignored.
        :return: The encoded sentences. If a single sentence was passed, a vector is returned.
        """
        was_single = False
        if isinstance(sentences, str):
            sentences = [sentences]
            was_single = True

        # Prepare all batches
        sentence_batches = list(self._batch(sentences, batch_size))
        total_batches = math.ceil(len(sentences) / batch_size)

        # Use joblib for multiprocessing if requested, and if we have enough sentences
        if use_multiprocessing and len(sentences) > multiprocessing_threshold:
            # Disable parallelism for tokenizers
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            results = ProgressParallel(n_jobs=-1, use_tqdm=show_progress_bar, total=total_batches)(
                delayed(self._encode_batch)(batch, max_length) for batch in sentence_batches
            )
            out_array = np.concatenate(results, axis=0)
        else:
            # Don't use multiprocessing
            out_arrays: list[np.ndarray] = []
            for batch in tqdm(
                sentence_batches,
                total=total_batches,
                disable=not show_progress_bar,
            ):
                out_arrays.append(self._encode_batch(batch, max_length))
            out_array = np.concatenate(out_arrays, axis=0)

        if was_single:
            return out_array[0]
        return out_array

    def _encode_batch(self, sentences: Sequence[str], max_length: int | None) -> np.ndarray:
        """Encode a batch of sentences."""
        ids = self.tokenize(sentences=sentences, max_length=max_length)
        out: list[np.ndarray] = []
        for id_list in ids:
            if id_list:
                out.append(self.embedding[id_list].mean(0))
            else:
                out.append(np.zeros(self.dim))

        out_array = np.stack(out)
        if self.normalize:
            norm = np.linalg.norm(out_array, axis=1, keepdims=True) + 1e-32
            out_array = out_array / norm

        return out_array

    @staticmethod
    def _batch(sentences: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
        """Batch the sentences into equal-sized."""
        return (sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size))

    def push_to_hub(
        self, repo_id: str, private: bool = False, token: str | None = None, subfolder: str | None = None
    ) -> None:
        """
        Push the model to the huggingface hub.

        NOTE: you need to pass a token if you are pushing a private model.

        :param repo_id: The repo id to push to.
        :param private: Whether the repo, if created is set to private.
            If the repo already exists, this doesn't change the visibility.
        :param token: The huggingface token to use.
        :param subfolder: The subfolder to push to.
        """
        from model2vec.hf_utils import push_folder_to_hub

        with TemporaryDirectory() as temp_dir:
            self.save_pretrained(temp_dir, model_name=repo_id)
            push_folder_to_hub(Path(temp_dir), subfolder=subfolder, repo_id=repo_id, private=private, token=token)

    @classmethod
    def load_local(cls: type[StaticModel], path: PathLike) -> StaticModel:
        """
        Loads a model from a local path.

        You should only use this code path if you are concerned with start-up time.
        Loading via the `from_pretrained` method is safer, and auto-downloads, but
        also means we import a whole bunch of huggingface code that we don't need.

        Additionally, huggingface will check the most recent version of the model,
        which can be slow.

        :param path: The path to load the model from. The path is a directory saved by the
            `save_pretrained` method.
        :return: A StaticModel
        :raises: ValueError if the path is not a directory.
        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")

        embeddings, tokenizer, config = load_local_model(path)

        return StaticModel(embeddings, tokenizer, config)
