from __future__ import annotations

import logging
import os
import re
from typing import Optional, cast

import numpy as np
from huggingface_hub import model_info
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from model2vec.distill.inference import PCADimType, create_embeddings, post_process_embeddings
from model2vec.distill.utils import select_optimal_device
from model2vec.model import StaticModel
from model2vec.quantization import DType, quantize_embeddings
from model2vec.tokenizer import clean_and_create_vocabulary, replace_vocabulary, turn_tokens_into_ids

logger = logging.getLogger(__name__)


def distill_from_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: list[str] | None = None,
    device: str | None = None,
    pca_dims: PCADimType = 256,
    apply_zipf: bool | None = None,
    sif_coefficient: float | None = 1e-4,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    quantize_to: DType | str = DType.Float16,
    use_subword: bool | None = None,
) -> StaticModel:
    """
    Distill a staticmodel from a sentence transformer.

    This function creates a set of embeddings from a sentence transformer. It does this by doing either
    a forward pass for all subword tokens in the tokenizer, or by doing a forward pass for all tokens in a passed vocabulary.

    If you pass through a vocabulary, we create a custom word tokenizer for that vocabulary.
    If you don't pass a vocabulary, we use the model's tokenizer directly.

    :param model: The model to use.
    :param tokenizer: The tokenizer to use.
    :param vocabulary: The vocabulary to use. If this is None, we use the model's vocabulary.
    :param device: The device to use.
    :param pca_dims: The number of components to use for PCA.
        If this is None, we don't apply PCA.
        If this is 'auto', we don't reduce dimensionality, but still apply PCA.
    :param apply_zipf: DEPRECATED: This parameter used to control whether Zipf is applied.
        Zipf weighting is now controlled by the sif_coefficient parameter. If this is set to None, no weighting is applied.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value > 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
        If the pattern is so general that it removes all tokens, we throw an error. If the pattern can't be compiled into a valid regex, we also throw an error.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param use_subword: DEPRECATED: If this is not set to None, we show a warning. It doesn't do anything.
    :return: A StaticModel
    :raises: ValueError if the vocabulary is empty after preprocessing.

    """
    if use_subword is not None:
        logger.warning(
            "The `use_subword` parameter is deprecated and will be removed in the next release. It doesn't do anything."
        )
    quantize_to = DType(quantize_to)
    backend_tokenizer = tokenizer.backend_tokenizer
    sif_coefficient, token_remove_regex = _validate_parameters(apply_zipf, sif_coefficient, token_remove_pattern)

    if vocabulary is None:
        vocabulary = []

    device = select_optimal_device(device)

    n_tokens_before = len(vocabulary)
    # Clean the vocabulary by removing duplicate tokens and tokens that are in the internal vocabulary.
    all_tokens, backend_tokenizer = clean_and_create_vocabulary(
        tokenizer, vocabulary, token_remove_regex=token_remove_regex
    )
    n_tokens_after = len([token for token in all_tokens if not token.is_internal])
    if n_tokens_before:
        logger.info(
            f"Adding {n_tokens_after} tokens to the vocabulary. Removed {n_tokens_before - n_tokens_after} tokens during preprocessing."
        )

    if not all_tokens:
        raise ValueError("The vocabulary is empty after preprocessing. Please check your token_remove_pattern.")

    unk_token = cast(Optional[str], tokenizer.special_tokens_map.get("unk_token"))
    pad_token = cast(Optional[str], tokenizer.special_tokens_map.get("pad_token"))

    # Weird if to satsify mypy
    if pad_token is None:
        if unk_token is not None:
            pad_token = unk_token
            logger.warning(
                "The pad token is not set. Setting it to the unk token. This is a workaround for models that don't have a pad token."
            )
        else:
            pad_token = unk_token or all_tokens[0].form
            logger.warning(
                "The pad token is not set. Setting it to the first token in the vocabulary. This is a workaround for models that don't have a pad token."
            )

    # Replace the vocabulary in the tokenizer with the new vocabulary.
    backend_tokenizer = replace_vocabulary(backend_tokenizer, all_tokens, unk_token=unk_token, pad_token=pad_token)

    logger.info(f"Creating embeddings for {len(all_tokens)} tokens")
    # Convert tokens to IDs
    token_ids = turn_tokens_into_ids(all_tokens, tokenizer, unk_token)

    # Create the embeddings
    embeddings = create_embeddings(
        tokenized=token_ids, model=model, device=device, pad_token_id=tokenizer.get_vocab()[pad_token]
    )

    # Post process the embeddings by applying PCA and Zipf weighting.
    embeddings = post_process_embeddings(np.asarray(embeddings), pca_dims, sif_coefficient=sif_coefficient)
    # Quantize the embeddings.
    embeddings = quantize_embeddings(embeddings, quantize_to)

    model_name = getattr(model, "name_or_path", "")

    config = {
        "model_type": "model2vec",
        "architectures": ["StaticModel"],
        "tokenizer_name": model_name,
        "apply_pca": pca_dims,
        "apply_zipf": apply_zipf,
        "sif_coefficient": sif_coefficient,
        "hidden_dim": embeddings.shape[1],
        "seq_length": 1000000,  # Set this to a high value since we don't have a sequence length limit.
        "normalize": True,
    }

    if os.path.exists(model_name):
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
        tokenizer=backend_tokenizer,
        config=config,
        base_model_name=model_name,
        language=language,
        normalize=True,
    )


def _validate_parameters(
    apply_zipf: bool | None,
    sif_coefficient: float | None,
    token_remove_pattern: str | None,
) -> tuple[float | None, re.Pattern | None]:
    """
    Validate the parameters passed to the distillation function.

    :param apply_zipf: DEPRECATED: This parameter used to control whether Zipf is applied.
        Zipf weighting is now controlled by the sif_coefficient parameter. If this is set to None, no weighting is applied.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value >= 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
    :return: The SIF coefficient to use.
    :raises: ValueError if the regex can't be compiled.

    """
    if apply_zipf is not None:
        logger.warning(
            "The `apply_zipf` parameter is deprecated and will be removed in the next release. "
            "Zipf weighting is applied based on the sif_coefficient parameter. If this is set to None, "
            "no weighting is applied."
        )
        if apply_zipf and sif_coefficient is None:
            logger.warning("You set apply_zipf to True, but sif_coefficient is None. Setting sif_coefficient to 1e-4.")
            sif_coefficient = 1e-4
        elif not apply_zipf:
            logger.warning("Because you set apply_zipf to False, we ignore the sif_coefficient parameter.")
            sif_coefficient = None

    if sif_coefficient is not None:
        if not 0 < sif_coefficient < 1.0:
            raise ValueError("SIF coefficient must be a value > 0 and < 1.0.")

    token_remove_regex: re.Pattern | None = None
    if token_remove_pattern is not None:
        try:
            token_remove_regex = re.compile(token_remove_pattern)
        except re.error as e:
            raise ValueError(f"Couldn't compile the regex pattern: {e}")

    return sif_coefficient, token_remove_regex


def distill(
    model_name: str,
    vocabulary: list[str] | None = None,
    device: str | None = None,
    pca_dims: PCADimType = 256,
    apply_zipf: bool | None = None,
    sif_coefficient: float | None = 1e-4,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    trust_remote_code: bool = False,
    quantize_to: DType | str = DType.Float16,
    use_subword: bool | None = None,
) -> StaticModel:
    """
    Distill a staticmodel from a sentence transformer.

    This function creates a set of embeddings from a sentence transformer. It does this by doing either
    a forward pass for all subword tokens in the tokenizer, or by doing a forward pass for all tokens in a passed vocabulary.

    If you pass through a vocabulary, we create a custom word tokenizer for that vocabulary.
    If you don't pass a vocabulary, we use the model's tokenizer directly.

    :param model_name: The model name to use. Any sentencetransformer compatible model works.
    :param vocabulary: The vocabulary to use. If this is None, we use the model's vocabulary.
    :param device: The device to use.
    :param pca_dims: The number of components to use for PCA.
        If this is None, we don't apply PCA.
        If this is 'auto', we don't reduce dimenionality, but still apply PCA.
    :param apply_zipf: DEPRECATED: This parameter used to control whether Zipf is applied.
        Zipf weighting is now controlled by the sif_coefficient parameter. If this is set to None, no weighting is applied.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value >= 0 and < 1.0. A value of 1e-4 is a good default.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
    :param trust_remote_code: Whether to trust the remote code. If this is False, we will only load components coming from `transformers`. If this is True, we will load all components.
    :param quantize_to: The data type to quantize to. Can be any of the DType enum members or their string equivalents.
    :param use_subword: DEPRECATED: If this is not set to None, we show a warning. It doesn't do anything.
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
        apply_zipf=apply_zipf,
        token_remove_pattern=token_remove_pattern,
        sif_coefficient=sif_coefficient,
        quantize_to=quantize_to,
        use_subword=use_subword,
    )
