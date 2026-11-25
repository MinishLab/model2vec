from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, TypeVar, cast, overload

import model2vec_rs
import numpy as np
import numpy.typing as npt
from tokenizers import Encoding, Tokenizer

from model2vec.quantization import DType, quantize_and_reduce_dim
from model2vec.types import _UNSET, StaticModelConfig, _UnsetType

PathLike = Path | str

logger = getLogger(__name__)

DEFAULT_MAX_LENGTH = 512
_DEFAULT_NORMALIZE = False


class StaticModel:
    def __init__(
        self,
        vectors: np.ndarray,
        tokenizer: Tokenizer,
        config: Mapping[str, Any] | None = None,
        normalize: bool = _DEFAULT_NORMALIZE,
        base_model_name: str | None = None,
        language: list[str] | None = None,
        weights: np.ndarray | None = None,
        token_mapping: np.ndarray | None = None,
        max_length: int | None = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Initialize the StaticModel.

        :param vectors: The vectors to use. The model stores read-only views of `vectors`, `weights` and
            `token_mapping`; assign new arrays to `embedding`, `weights` or `token_mapping` to change them.
        :param tokenizer: The Transformers tokenizer to use. Stored as a copy. If you change `model.tokenizer`
            in place, assign it again (`model.tokenizer = model.tokenizer`) for `encode` to use the changes.
        :param config: Any metadata config. Stored as a copy, so mutating it afterwards does not affect
            the model, and vice versa.
        :param normalize: Whether to normalize the embeddings. Defaults to False. This value is authoritative:
            it is written to `config["normalize"]`, and any conflicting value already in `config` is overridden.
        :param base_model_name: The used base model name. Used for creating a model card.
        :param language: The language of the model. Used for creating a model card.
        :param weights: The weights to use for the embeddings. If None, no weights are used.
            We always assume the norm of the embeddings is an implicit weight anyway.
            This is only used for models that have undergone vocabulary quantization.
        :param token_mapping: A mapping from token ids to indices in the vectors.
            If None, we don't remap the tokens during inference.
            This is only used for models that have undergone vocabulary quantization.
        :param max_length: The default maximum sequence length (in tokens) used by `encode` when its own
            `max_length` argument is not passed. Defaults to 512; pass None to disable truncation. This value
            is authoritative: it is written to `config["max_length"]`, and any conflicting value already in
            `config` is overridden.
        :raises ValueError: if the number of tokens does not match the number of vectors.
        """
        super().__init__()
        tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
        self.tokens = tokens

        if token_mapping is None and len(vectors) != len(tokens):
            raise ValueError(
                f"Number of tokens ({len(tokens)}) does not match number of vectors ({len(vectors)}). "
                "Please provide a token mapping or ensure the number of tokens matches the number of vectors."
            )

        self.embedding = vectors
        self.weights = weights
        self.token_mapping = token_mapping

        self.tokenizer = copy.deepcopy(tokenizer)
        _disable_padding(self.tokenizer)
        self.unk_token_id = _get_unk_token_id(self.tokenizer)

        self.median_token_length = int(np.median([len(token) for token in self.tokens]))
        self.config: StaticModelConfig = cast(StaticModelConfig, {**config}) if config is not None else {}
        self.base_model_name = base_model_name
        self.language = language
        self.max_length = max_length
        self.normalize = normalize

    def __getstate__(self) -> dict[str, Any]:
        """Get the state for pickling, without the Rust backend."""
        state = self.__dict__.copy()
        state.pop("_rust_backend", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the state from pickling, keeping the arrays read-only."""
        for name in ("embedding", "weights", "token_mapping"):
            state[f"_{name}"] = _read_only(state.pop(f"_{name}", state.pop(name, None)))
        state["_tokenizer"] = state.pop("_tokenizer", state.pop("tokenizer", None))
        self.__dict__.update(state)

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer. Assign it again after changing it in place, so that `encode` uses the changes."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer) -> None:
        self._tokenizer = value
        self.__dict__.pop("_rust_backend", None)

    @property
    def embedding(self) -> np.ndarray:
        """The read-only embedding matrix."""
        return self._embedding

    @embedding.setter
    def embedding(self, value: np.ndarray) -> None:
        self._embedding = _read_only(value)

    @property
    def weights(self) -> np.ndarray | None:
        """The read-only token weights, or None."""
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray | None) -> None:
        self._weights = _read_only(value)

    @property
    def token_mapping(self) -> np.ndarray | None:
        """The read-only mapping from token ids to rows of `embedding`, or None."""
        return self._token_mapping

    @token_mapping.setter
    def token_mapping(self, value: np.ndarray | None) -> None:
        self._token_mapping = _read_only(value)

    def _get_rust_backend(self) -> model2vec_rs.StaticModel:
        """Get the Rust backend used by `encode`, rebuilding it if the vectors or tokenizer were assigned.

        Building the backend replaces `embedding` and `weights` with read-only views of the backend's memory
        when their dtypes match.

        :return: The Rust backend.
        """
        key = (self.embedding, self.weights, self.token_mapping)
        cached = self.__dict__.get("_rust_backend")
        if cached is not None and all(old is new for old, new in zip(cached[0], key)):
            return cached[1]
        embedding: npt.NDArray[np.float32] | npt.NDArray[np.int8]
        if self.embedding.dtype == np.int8:
            embedding = np.ascontiguousarray(self.embedding, dtype=np.int8)
        else:
            embedding = np.ascontiguousarray(self.embedding, dtype=np.float32)
        backend = model2vec_rs.StaticModel.from_arrays(
            self.tokenizer.to_str(),
            embedding,
            normalize=False,
            weights=None if self.weights is None else np.ascontiguousarray(self.weights, dtype=np.float32),
            token_mapping=(
                None if self.token_mapping is None else np.ascontiguousarray(self.token_mapping, dtype=np.int64)
            ),
        )
        shared_embedding, shared_weights = backend.embeddings, backend.weights
        if shared_embedding is not None and shared_embedding.dtype == self.embedding.dtype:
            self.embedding = shared_embedding
        if shared_weights is not None and self.weights is not None and self.weights.dtype == shared_weights.dtype:
            self.weights = shared_weights
        self._rust_backend = ((self.embedding, self.weights, self.token_mapping), backend)
        return backend

    @property
    def dim(self) -> int:
        """Get the dimension of the model."""
        return self.embedding.shape[1]

    @property
    def normalize(self) -> bool:
        """Get the normalize value.

        :return: The normalize value.
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        """Update the config."""
        config_normalize = self.config.get("normalize")
        self._normalize = value
        if config_normalize is not None and value != config_normalize:
            logger.warning(
                f"Set normalization to `{value}`, which does not match config value `{config_normalize}`. Updating config."
            )
        self.config["normalize"] = value

    @property
    def max_length(self) -> int | None:
        """Get the max_length value.

        :return: The max_length value.
        """
        return self._max_length

    @max_length.setter
    def max_length(self, value: int | None) -> None:
        """Update the config."""
        config_max_length = self.config.get("max_length")
        self._max_length = value
        if config_max_length is not None and value != config_max_length:
            logger.warning(
                f"Set max_length to `{value}`, which does not match config value `{config_max_length}`. Updating config."
            )
        self.config["max_length"] = value
        self._set_max_length_in_tokenizer(value)

    @property
    def embedding_dtype(self) -> str:
        """Get the dtype (precision) of the embedding matrix."""
        return np.dtype(self.embedding.dtype).name

    @property
    def vocabulary_quantization(self) -> int | None:
        """Get the number of clusters used for vocabulary quantization, if applicable."""
        is_quantized = (self.token_mapping is not None) or (len(self.embedding) != len(self.tokens))
        return int(self.embedding.shape[0]) if is_quantized else None

    def save_pretrained(self, path: PathLike, model_name: str | None = None, subfolder: str | None = None) -> None:
        """Save the pretrained model.

        :param path: The path to save to.
        :param model_name: The model name to use in the Model Card.
        :param subfolder: The subfolder to save to.
        """
        from model2vec.persistence import save_pretrained

        save_pretrained(
            folder_path=Path(path),
            embeddings=self.embedding,
            tokenizer=self.tokenizer,
            config=self.config,
            base_model_name=self.base_model_name,
            language=self.language,
            model_name=model_name,
            subfolder=subfolder,
            weights=self.weights,
            mapping=self.token_mapping,
        )

    def tokenize(self, sentences: Sequence[str]) -> list[list[int]]:
        """Tokenize a list of sentences.

        :param sentences: The sentences to tokenize.
        :return: A list of list of tokens.
        """
        encodings: list[Encoding] = self.tokenizer.encode_batch_fast(sentences, add_special_tokens=False)

        encodings_ids = [encoding.ids for encoding in encodings]

        if self.unk_token_id is not None:
            # NOTE: Remove the unknown token: necessary for word-level models.
            encodings_ids = [
                [token_id for token_id in token_ids if token_id != self.unk_token_id] for token_ids in encodings_ids
            ]

        return encodings_ids

    @classmethod
    def from_pretrained(
        cls: type[T],
        path: PathLike,
        token: str | None = None,
        normalize: bool | None = None,
        subfolder: str | None = None,
        quantize_to: str | DType | None = None,
        dimensionality: int | None = None,
        vocabulary_quantization: int | None = None,
        max_length: int | None | _UnsetType = _UNSET,
        force_download: bool = True,
    ) -> T:
        """Load a StaticModel from a local path or huggingface hub path.

        NOTE: if you load a private model from the huggingface hub, you need to pass a token.

        :param path: The path to load your static model from.
        :param token: The huggingface token to use.
        :param normalize: Whether to normalize the embeddings. If not passed, the value from the model's config
            is used, falling back to False if the config does not specify one.
        :param subfolder: The subfolder to load from.
        :param quantize_to: The dtype to quantize the model to. If None, no quantization is done.
            If a string is passed, it is converted to a DType.
        :param dimensionality: The dimensionality of the model. If this is None, use the dimensionality of the model.
            This is useful if you want to load a model with a lower dimensionality.
            Note that this only applies if you have trained your model using mrl or PCA.
        :param vocabulary_quantization: The number of clusters to use for vocabulary quantization.
        :param max_length: The default maximum sequence length (in tokens) for `encode`. If not passed, the
            value from the model's config is used, falling back to 512 if the config does not specify one.
            Pass None to disable truncation.
        :param force_download: Whether to force the download of the model. If False, the model is only downloaded if it is not
            already present in the cache.
        :return: A StaticModel.
        """
        return _loading_helper(
            cls=cls,
            max_length=max_length,
            path=path,
            token=token,
            vocabulary_quantization=vocabulary_quantization,
            quantize_to=quantize_to,
            dimensionality=dimensionality,
            normalize=normalize,
            subfolder=subfolder,
            force_download=force_download,
        )

    def _set_max_length_in_tokenizer(self, max_length: int | None) -> None:
        """Sets the max length in the tokenizer."""
        if max_length is None:
            self.tokenizer.no_truncation()
        else:
            self.tokenizer.enable_truncation(max_length)

    @overload
    def encode_as_sequence(self, sentences: str, *, max_length: int | None = None, **kwargs: Any) -> np.ndarray: ...

    @overload
    def encode_as_sequence(
        self, sentences: list[str], *, max_length: int | None = None, **kwargs: Any
    ) -> list[np.ndarray]: ...

    def encode_as_sequence(
        self, sentences: str | list[str], *, max_length: int | None = None, **kwargs: Any
    ) -> list[np.ndarray] | np.ndarray:
        """Encode a list of sentences as a list of numpy arrays of tokens.

        This is useful if you want to use the tokens for further processing, or if you want to do sequence
        modeling.
        Note that if you just want the mean, you should use the `encode` method, which is much faster.
        Sentences that do not contain any tokens will be turned into an empty array.
        Unlike `encode`, this method ignores the model's `max_length`: truncation is only applied if you
        explicitly pass `max_length` here.

        NOTE: the input type is currently underspecified. The actual input type is `Sequence[str] | str`, but this
            is not possible to implement in python typing currently.

        :param sentences: The list of sentences to encode.
        :param max_length: The maximum length of the sentences. Any tokens beyond this length will be truncated.
            If this is None, no truncation is done. Note that this is independent of the model's `max_length`.
        :param **kwargs: Any additional arguments. These are ignored. The removed `batch_size`,
            `show_progress_bar`, `use_multiprocessing` and `multiprocessing_threshold` raise a DeprecationWarning.
        :return: The encoded sentences with an embedding per token.
        """
        _warn_removed_arguments(kwargs)
        was_single = isinstance(sentences, str)
        sentence_list: list[str] = [sentences] if isinstance(sentences, str) else list(sentences)
        if max_length is not None:
            m = max_length * self.median_token_length
            sentence_list = [sentence[:m] for sentence in sentence_list]

        self._set_max_length_in_tokenizer(max_length)
        try:
            ids = self.tokenize(sentence_list)
        finally:
            self._set_max_length_in_tokenizer(self.max_length)

        out = [self._encode_helper(id_list) if id_list else np.zeros((0, self.dim)) for id_list in ids]
        if was_single:
            return out[0]
        return out

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        max_length: int | None | _UnsetType = _UNSET,
        normalize: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a list of sentences.

        This function encodes a list of sentences by averaging the word embeddings of the tokens in the sentence.

        NOTE: the return type is currently underspecified. In the case of a single string, this returns a 1D array,
            but in the case of a list of strings, this returns a 2D array. Not possible to implement in numpy currently.

        :param sentences: The list of sentences to encode. You can also pass a single sentence.
        :param max_length: The maximum length of the sentences. Any tokens beyond this length will be truncated.
            If this is None, no truncation is done. If not passed, the model's `max_length` is used.
        :param normalize: Whether to normalize the resulting embeddings. If not passed, the model's `normalize`
            setting is used.
        :param **kwargs: Any additional arguments. These are ignored. The removed `batch_size`,
            `show_progress_bar`, `use_multiprocessing` and `multiprocessing_threshold` raise a DeprecationWarning.
        :return: The encoded sentences. If a single sentence was passed, a vector is returned.
        """
        _warn_removed_arguments(kwargs)
        if isinstance(max_length, _UnsetType):
            max_length = self.max_length
        if normalize is None:
            normalize = self.normalize

        out = self._get_rust_backend().encode(sentences, max_length=max_length, normalize=normalize)
        if self.embedding.dtype != np.int8:
            out = out.astype(self.embedding.dtype, copy=False)
        return out

    def _encode_helper(self, id_list: list[int]) -> np.ndarray:
        """Helper function to encode a list of ids.

        It retrieves the embeddings for the given list of ids, applying weights if available.

        :param id_list: A list of token ids.
        :return: The embeddings for the given ids, as a sequence of vectors.
        """
        id_list_remapped: list[int] | np.ndarray
        if self.token_mapping is None:
            id_list_remapped = id_list
        else:
            id_list_remapped = self.token_mapping[id_list]
        emb = self.embedding[id_list_remapped]
        if self.weights is not None:
            emb = emb * self.weights[id_list][:, None]

        return emb

    def push_to_hub(
        self, repo_id: str, private: bool = False, token: str | None = None, subfolder: str | None = None
    ) -> None:
        """Push the model to the huggingface hub.

        NOTE: you need to pass a token if you are pushing a private model.

        :param repo_id: The repo id to push to.
        :param private: Whether the repo, if created is set to private.
            If the repo already exists, this doesn't change the visibility.
        :param token: The huggingface token to use.
        :param subfolder: The subfolder to push to.
        """
        from model2vec.persistence import push_folder_to_hub

        with TemporaryDirectory() as temp_dir:
            self.save_pretrained(temp_dir, model_name=repo_id)
            push_folder_to_hub(Path(temp_dir), subfolder=subfolder, repo_id=repo_id, private=private, token=token)


def quantize_model(
    model: T,
    vocabulary_quantization: int | None = None,
    quantize_to: str | DType | None = None,
    dimensionality: int | None = None,
) -> T:
    """Quantize the model to a lower precision and possibly lower dimensionality.

    :param model: The model to quantize.
    :param vocabulary_quantization: The number of clusters to use for quantization.
    :param quantize_to: The dtype to quantize the model to.
    :param dimensionality: The desired dimensionality of the model.
        This needs to be < than the current model dimensionality.
    :return: A new StaticModel with the quantized embeddings.
    :raises ValueError: if the model is already quantized.
    """
    token_mapping: np.ndarray | None
    weights: np.ndarray | None
    if vocabulary_quantization is not None:
        from model2vec.vocabulary_quantization import quantize_vocabulary

        if len(model.tokens) != len(model.embedding):
            raise ValueError("Model already has been vocabulary quantized, cannot quantize again.")

        embeddings, token_mapping, weights = quantize_vocabulary(
            n_clusters=vocabulary_quantization, weights=model.weights, embeddings=model.embedding
        )
    else:
        embeddings = model.embedding
        token_mapping = model.token_mapping
        weights = model.weights
    if quantize_to is not None or dimensionality is not None:
        embeddings = quantize_and_reduce_dim(
            embeddings=embeddings,
            quantize_to=quantize_to,
            dimensionality=dimensionality,
        )

    return type(model)(
        vectors=embeddings,
        tokenizer=model.tokenizer,
        config=model.config,
        weights=weights,
        token_mapping=token_mapping,
        normalize=model.normalize,
        base_model_name=model.base_model_name,
        language=model.language,
        max_length=model.max_length,
    )


def _loading_helper(
    cls: type[T],
    path: PathLike,
    token: str | None,
    vocabulary_quantization: int | None,
    quantize_to: str | DType | None,
    dimensionality: int | None,
    normalize: bool | None,
    subfolder: str | None,
    force_download: bool,
    max_length: int | None | _UnsetType,
) -> T:
    """Helper function to load a model from a directory."""
    from model2vec.persistence import load_pretrained

    embeddings, tokenizer, config, metadata, weights, mapping = load_pretrained(
        folder_or_repo_path=path,
        token=token,
        subfolder=subfolder,
        force_download=force_download,
    )

    normalize = normalize if normalize is not None else config.get("normalize", _DEFAULT_NORMALIZE)
    resolved_max_length = (
        config.get("max_length", DEFAULT_MAX_LENGTH) if isinstance(max_length, _UnsetType) else max_length
    )

    model = cls(
        vectors=embeddings,
        tokenizer=tokenizer,
        weights=weights,
        token_mapping=mapping,
        config=config,
        normalize=normalize,
        base_model_name=metadata.get("base_model"),
        language=metadata.get("language"),
        max_length=resolved_max_length,
    )

    # If no quantization or dimensionality reduction is requested,
    # return the model as is.
    if not any([vocabulary_quantization, quantize_to, dimensionality]):
        return model

    return quantize_model(
        model=model,
        vocabulary_quantization=vocabulary_quantization,
        quantize_to=quantize_to,
        dimensionality=dimensionality,
    )


def _disable_padding(tokenizer: Tokenizer) -> None:
    """Stop the tokenizer from padding, while keeping its pad token."""
    padding = tokenizer.padding
    if padding is not None:
        tokenizer.enable_padding(
            pad_id=padding["pad_id"], pad_token=padding["pad_token"], pad_type_id=padding["pad_type_id"], length=0
        )


def _get_unk_token_id(tokenizer: Tokenizer) -> int | None:
    """Get the unk token id."""
    model = tokenizer.model
    # Wordpiece + BPE + Word level
    if hasattr(model, "unk_token"):
        token = model.unk_token
        if token is None:
            return None
        return tokenizer.token_to_id(token)
    # Unigram
    return json.loads(tokenizer.to_str())["model"].get("unk_id")


_REMOVED_ARGUMENTS = ("batch_size", "show_progress_bar", "use_multiprocessing", "multiprocessing_threshold")


def _warn_removed_arguments(kwargs: Mapping[str, Any], stacklevel: int = 3) -> None:
    """Warn about removed encoding arguments in `kwargs`.

    :param kwargs: The keyword arguments a method received.
    :param stacklevel: The stack level of the warning, counted from this function.
    """
    removed = [name for name in _REMOVED_ARGUMENTS if name in kwargs]
    if removed:
        warnings.warn(
            f"{', '.join(f'`{name}`' for name in removed)} no longer have any effect and will be removed "
            "in a future version.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )


@overload
def _read_only(array: np.ndarray) -> np.ndarray: ...


@overload
def _read_only(array: np.ndarray | None) -> np.ndarray | None: ...


def _read_only(array: np.ndarray | None) -> np.ndarray | None:
    """Return a read-only view of an array, leaving the array itself writable."""
    if array is None or not array.flags.writeable:
        return array
    view = array.view()
    view.setflags(write=False)
    return view


T = TypeVar("T", bound=StaticModel)
