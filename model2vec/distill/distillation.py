from __future__ import annotations

import logging
import os
import re
from typing import Sequence, cast

import numpy as np
from huggingface_hub.hf_api import model_info
from skeletoken import TokenizerModel
from skeletoken.external.transformers import reshape_embeddings
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel

from model2vec.distill.inference import PCADimType, PoolingMode, apply_pca, compute_weights, create_embeddings
from model2vec.distill.utils import select_optimal_device
from model2vec.model import StaticModel
from model2vec.quantization import DType, quantize_embeddings
from model2vec.tokenizer import clean_and_create_vocabulary, turn_tokens_into_ids
from model2vec.types import StaticModelConfig
from model2vec.vocabulary_quantization import quantize_vocabulary

logger = logging.getLogger(__name__)


def distill_from_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: Sequence[str] | None = None,
    device: str | None = None,
    pca_dims: PCADimType = 256,
    sif_coefficient: float | None = 1e-4,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    quantize_to: DType | str = DType.Float16,
    vocabulary_quantization: int | None = None,
    pooling: PoolingMode | str = PoolingMode.MEAN,
) -> StaticModel:
    """Distill a staticmodel from a sentence transformer.

    This function creates a set of embeddings from a sentence transformer. It does this by doing either
    a forward pass for all subword tokens in the tokenizer, or by doing a forward pass for all tokens in a passed
    vocabulary.

    If you pass through a vocabulary, we create a custom word tokenizer for that vocabulary.
    If you don't pass a vocabulary, we use the model's tokenizer directly.

    :param model: The model to use.
    :param tokenizer: The tokenizer to use.
    :param vocabulary: The vocabulary to use. If this is None, we use the model's vocabulary.
    :param device: The device to use.
    :param pca_dims: The number of components to use for PCA.
        If this is None, we don't apply PCA.
        If this is 'auto', we don't reduce dimensionality, but still apply PCA.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value > 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to
        this regex pattern will be removed from the vocabulary.
        If the pattern is so general that it removes all tokens, we throw an error. If the pattern can't be compiled
        into a valid regex, we also throw an error.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param vocabulary_quantization: The number of clusters to use for vocabulary quantization. If this is None, no
         quantization is performed.
    :param pooling: The pooling mode to use for creating embeddings. Can be one of:
        'mean' (default): mean over all tokens. Robust and works well in most cases.
        'last': use the last token's hidden state (often the [EOS] token). Common for decoder-style models.
        'first': use the first token's hidden state ([CLS] token in BERT-style models).
        'pooler': use the pooler output (if available). This is often a non-linear projection of the [CLS] token.
    :return: A StaticModel.

    """
    quantize_to = DType(quantize_to)
    sif_coefficient, token_remove_regex = _validate_parameters(sif_coefficient, token_remove_pattern)

    device = select_optimal_device(device)
    original_tokenizer_model, tokenizer_model = _prepare_tokenizer(tokenizer, vocabulary, token_remove_regex)

    # Turn all _new_ tokens into ids using the original tokenizer
    token_ids = turn_tokens_into_ids(tokenizer_model.sorted_vocabulary, original_tokenizer_model)

    # Reshape the transformer
    model = reshape_embeddings(model, original_tokenizer_model)

    # Create the embeddings using the ids from the original tokenizer.
    embeddings = create_embeddings(
        tokenized=token_ids,
        model=model,
        device=device,
        pad_token_id=original_tokenizer_model.pad_token_id or 0,
        pooling=pooling,
    )

    # Apply quantization
    if vocabulary_quantization is not None:
        weights = compute_weights(len(embeddings), sif_coefficient=sif_coefficient)
        embeddings, token_mapping, weights = quantize_vocabulary(
            n_clusters=vocabulary_quantization, weights=weights, embeddings=embeddings
        )
        embeddings = apply_pca(embeddings, pca_dims)
    else:
        # Post-process the embeddings.
        weights = compute_weights(len(embeddings), sif_coefficient=sif_coefficient)
        embeddings = apply_pca(embeddings, pca_dims)
        embeddings = embeddings * weights[:, None]
        weights = None
        token_mapping = None
    # Quantize the embeddings.
    embeddings = quantize_embeddings(embeddings, quantize_to)

    model_name = getattr(model, "name_or_path", "")

    config: StaticModelConfig = {
        "model_type": "model2vec",
        "architectures": ["StaticModel"],
        "tokenizer_name": model_name,
        "apply_pca": pca_dims,
        "sif_coefficient": sif_coefficient,
        "hidden_dim": embeddings.shape[1],
        "seq_length": 1000000,  # Set this to a high value since we don't have a sequence length limit.
        "normalize": True,
        "pooling": pooling,
    }

    return _create_static_model(
        embeddings=embeddings,
        weights=weights,
        token_mapping=token_mapping,
        tokenizer_model=tokenizer_model,
        config=config,
        model_name=model_name,
    )


def random_init(
    model_name: str,
    vocabulary: Sequence[str] | None = None,
    dim: int = 256,
    sif_coefficient: float | None = 1e-4,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    quantize_to: DType | str = DType.Float16,
    seed: int = 42,
    trust_remote_code: bool = False,
) -> StaticModel:
    """Create a staticmodel with random embeddings from a tokenizer.

    The tokenizer and vocabulary are processed in the same way as in `distill_from_model`, but instead of
    computing embeddings with a model, each token gets an embedding drawn from N(0, 1).

    :param model_name: The name or path of the model whose tokenizer to load.
    :param vocabulary: The vocabulary to use. If this is None, we use the tokenizer's vocabulary.
    :param dim: The dimensionality of the embeddings.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value > 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to
        this regex pattern will be removed from the vocabulary.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param seed: The random seed used to draw the embeddings.
    :param trust_remote_code: Whether to trust remote code when loading the tokenizer.
    :return: A StaticModel.

    """
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True),
    )
    return init_from_tokenizer(
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        dim=dim,
        sif_coefficient=sif_coefficient,
        token_remove_pattern=token_remove_pattern,
        quantize_to=quantize_to,
        seed=seed,
    )


def init_from_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: Sequence[str] | None = None,
    dim: int = 256,
    sif_coefficient: float | None = 1e-4,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    quantize_to: DType | str = DType.Float16,
    seed: int = 42,
) -> StaticModel:
    """Create a staticmodel with random embeddings from a tokenizer.

    The tokenizer and vocabulary are processed in the same way as in `distill_from_model`, but instead of
    computing embeddings with a model, each token gets an embedding drawn from N(0, 1).

    :param tokenizer: The tokenizer to use.
    :param vocabulary: The vocabulary to use. If this is None, we use the tokenizer's vocabulary.
    :param dim: The dimensionality of the embeddings.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value > 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to
        this regex pattern will be removed from the vocabulary.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param seed: The random seed used to draw the embeddings.
    :return: A StaticModel.
    :raises ValueError: if the parameters are invalid or the vocabulary is empty after preprocessing.

    """
    if dim <= 0:
        raise ValueError(f"dim must be a positive integer, got {dim}.")
    quantize_to = DType(quantize_to)
    sif_coefficient, token_remove_regex = _validate_parameters(sif_coefficient, token_remove_pattern)

    _, tokenizer_model = _prepare_tokenizer(tokenizer, vocabulary, token_remove_regex)

    embeddings = np.random.default_rng(seed).standard_normal((tokenizer_model.vocabulary_size, dim))
    weights = compute_weights(len(embeddings), sif_coefficient=sif_coefficient)
    embeddings = quantize_embeddings(embeddings * weights[:, None], quantize_to)

    tokenizer_name = tokenizer.name_or_path

    config: StaticModelConfig = {
        "model_type": "model2vec",
        "architectures": ["StaticModel"],
        "tokenizer_name": tokenizer_name,
        "apply_pca": None,
        "sif_coefficient": sif_coefficient,
        "hidden_dim": embeddings.shape[1],
        "seq_length": 1000000,
        "normalize": True,
        "pooling": PoolingMode.RANDOM,
    }

    return _create_static_model(
        embeddings=embeddings,
        weights=None,
        token_mapping=None,
        tokenizer_model=tokenizer_model,
        config=config,
        model_name=tokenizer_name,
    )


def _prepare_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: Sequence[str] | None,
    token_remove_regex: re.Pattern[str] | None,
) -> tuple[TokenizerModel, TokenizerModel]:
    """Create the original tokenizer model and the new tokenizer model.

    :param tokenizer: The original tokenizer.
    :param vocabulary: The vocabulary to add. If this is None, no tokens are added.
    :param token_remove_regex: Tokens matching this regex are removed from the vocabulary.
    :return: The original tokenizer model and the new tokenizer model.
    :raises ValueError: if the vocabulary is empty after preprocessing.

    """
    original_tokenizer_model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    original_tokenizer_model = original_tokenizer_model.prune_added_tokens()

    # Clean the vocabulary by removing duplicate tokens and tokens that are in the internal vocabulary.
    # Copy the original tokenizer model.
    tokenizer_model = original_tokenizer_model.deep_copy()
    if tokenizer_model.adds_prefix_space is not None:
        tokenizer_model.adds_prefix_space = True

    vocabulary = vocabulary if vocabulary is not None else []
    # Create the vocabulary in the new tokenizer.
    tokenizer_model = clean_and_create_vocabulary(tokenizer_model, vocabulary, token_remove_regex=token_remove_regex)

    if tokenizer_model.vocabulary_size == 0:
        raise ValueError("The vocabulary is empty after preprocessing. Please check your token_remove_pattern.")

    return original_tokenizer_model, tokenizer_model


def _create_static_model(
    embeddings: np.ndarray,
    weights: np.ndarray | None,
    token_mapping: np.ndarray | None,
    tokenizer_model: TokenizerModel,
    config: StaticModelConfig,
    model_name: str,
) -> StaticModel:
    """Create a StaticModel, looking up the language of the base model on the Hugging Face Hub.

    :param embeddings: The embeddings.
    :param weights: The token weights, if any.
    :param token_mapping: The mapping from tokens to embeddings, if any.
    :param tokenizer_model: The tokenizer model.
    :param config: The config of the model.
    :param model_name: The name or path of the base model.
    :return: A StaticModel.

    """
    if not model_name:
        language = None
    elif os.path.exists(model_name):
        # Using a local model. Get the model name from the path.
        model_name = os.path.basename(model_name)
        language = None
    else:
        # Get the language from the model card.
        try:
            info = model_info(model_name)
            language = info.cardData.get("language", None) if info.cardData is not None else None
        except Exception as e:
            # NOTE: bare except because there's many reasons this can fail.
            logger.warning(f"Couldn't get the model info from the Hugging Face Hub: {e}. Setting language to None.")
            language = None

    return StaticModel(
        vectors=embeddings,
        weights=weights,
        token_mapping=token_mapping,
        tokenizer=tokenizer_model.to_tokenizer(),
        config=config,
        base_model_name=model_name,
        language=language,
        normalize=True,
    )


def _validate_parameters(
    sif_coefficient: float | None,
    token_remove_pattern: str | None,
) -> tuple[float | None, re.Pattern[str] | None]:
    """Validate the parameters passed to the distillation function.

    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value > 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to
        this regex pattern will be removed from the vocabulary.
    :return: The SIF coefficient to use.
    :raises ValueError: if the regex can't be compiled.

    """
    if sif_coefficient is not None:
        if not 0 < sif_coefficient < 1.0:
            raise ValueError("SIF coefficient must be a value > 0 and < 1.0.")

    token_remove_regex: re.Pattern[str] | None = None
    if token_remove_pattern is not None:
        try:
            token_remove_regex = re.compile(token_remove_pattern)
        except re.error as e:
            raise ValueError(f"Couldn't compile the regex pattern: {e}") from e

    return sif_coefficient, token_remove_regex


def distill(
    model_name: str,
    vocabulary: list[str] | None = None,
    device: str | None = None,
    pca_dims: PCADimType = 256,
    sif_coefficient: float | None = 1e-4,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    trust_remote_code: bool = False,
    quantize_to: DType | str = DType.Float16,
    vocabulary_quantization: int | None = None,
    pooling: PoolingMode | str = PoolingMode.MEAN,
) -> StaticModel:
    """Distill a staticmodel from a sentence transformer.

    This function creates a set of embeddings from a sentence transformer. It does this by doing either
    a forward pass for all subword tokens in the tokenizer, or by doing a forward pass for all tokens in a passed
    vocabulary.

    If you pass through a vocabulary, we create a custom word tokenizer for that vocabulary.
    If you don't pass a vocabulary, we use the model's tokenizer directly.

    :param model_name: The model name to use. Any sentencetransformer compatible model works.
    :param vocabulary: The vocabulary to use. If this is None, we use the model's vocabulary.
    :param device: The device to use.
    :param pca_dims: The number of components to use for PCA.
        If this is None, we don't apply PCA.
        If this is 'auto', we don't reduce dimenionality, but still apply PCA.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value > 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to
        this regex pattern will be removed from the vocabulary.
    :param trust_remote_code: Whether to trust the remote code. If this is False, we will only load components coming
        from `transformers`. If this is True, we will load all components.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param vocabulary_quantization: The number of clusters to use for vocabulary quantization. If this is None, no
        quantization is performed.
    :param pooling: The pooling mode to use for creating embeddings. Can be one of:
        'mean' (default): mean over all tokens. Robust and works well in most cases.
        'last': use the last token's hidden state (often the [EOS] token). Common for decoder-style models.
        'first': use the first token's hidden state ([CLS] token in BERT-style models).
        'pooler': use the pooler output (if available). This is often a non-linear projection of the [CLS] token.
    :return: A StaticModel

    """
    model: PreTrainedModel = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True),
    )

    return distill_from_model(
        model=model,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        device=device,
        pca_dims=pca_dims,
        token_remove_pattern=token_remove_pattern,
        sif_coefficient=sif_coefficient,
        quantize_to=quantize_to,
        vocabulary_quantization=vocabulary_quantization,
        pooling=pooling,
    )
