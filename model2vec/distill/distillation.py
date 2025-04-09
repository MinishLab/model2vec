from __future__ import annotations

import logging
import os
import re
from typing import Literal, Union

import numpy as np
from huggingface_hub import model_info
from sklearn.decomposition import PCA
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from model2vec.distill.inference import create_embeddings
from model2vec.distill.tokenizer import replace_vocabulary
from model2vec.distill.utils import select_optimal_device
from model2vec.model import StaticModel

try:
    # For huggingface_hub>=0.25.0
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    # For huggingface_hub<0.25.0
    from huggingface_hub.utils._errors import RepositoryNotFoundError


logger = logging.getLogger(__name__)


PCADimType = Union[int, None, float, Literal["auto"]]


def distill_from_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: list[str] | None = None,
    device: str | None = None,
    pca_dims: PCADimType = 256,
    apply_zipf: bool | None = None,
    sif_coefficient: float | None = 1e-4,
    use_subword: bool = True,
    token_remove_pattern: str | None = r"\[unused\d+\]",
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
    :param use_subword: Whether to keep subword tokens in the vocabulary. If this is False, you must pass a vocabulary, and the returned tokenizer will only detect full words.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
        If the pattern is so general that it removes all tokens, we throw an error. If the pattern can't be compiled into a valid regex, we also throw an error.
    :return: A StaticModel

    """
    backend_tokenizer = tokenizer.backend_tokenizer
    sif_coefficient, token_remove_regex = _validate_parameters(
        vocabulary, apply_zipf, sif_coefficient, use_subword, token_remove_pattern
    )

    if vocabulary is None:
        vocabulary = []

    device = select_optimal_device(device)
    # Make a base list of tokens.
    subword_vocab: dict[str, int] = tokenizer.get_vocab()
    subword_tokens: list[str] = [k for k, _ in sorted(subword_vocab.items(), key=lambda x: x[1])]

    n_tokens_before = len(vocabulary)
    # Clean the vocabulary by removing duplicate tokens and tokens that are in the subword vocabulary.
    cleaned_vocabulary = _clean_vocabulary(tokenizer.backend_tokenizer, vocabulary, subword_tokens)
    n_tokens_after = len(cleaned_vocabulary)
    logger.info(
        f"Adding {n_tokens_after} tokens to the vocabulary. Removed {n_tokens_before - n_tokens_after} tokens during preprocessing."
    )

    # Create the embeddings.
    all_tokens, embeddings = create_embeddings(
        model=model,
        tokenizer=tokenizer,
        tokens=cleaned_vocabulary,
        device=device,
        use_subword=use_subword,
        token_remove_regex=token_remove_regex,
    )

    unk_token = tokenizer.special_tokens_map.get("unk_token")
    pad_token = tokenizer.special_tokens_map.get("pad_token")
    # Add the cleaned vocabulary to the tokenizer.
    backend_tokenizer = replace_vocabulary(backend_tokenizer, all_tokens, unk_token=unk_token, pad_token=pad_token)

    # Post process the embeddings by applying PCA and Zipf weighting.
    embeddings = _post_process_embeddings(np.asarray(embeddings), pca_dims, sif_coefficient=sif_coefficient)

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
            language = info.cardData.get("language", None)
        except RepositoryNotFoundError:
            logger.info("No model info found for the model. Setting language to None.")
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
    vocabulary: list[str] | None,
    apply_zipf: bool | None,
    sif_coefficient: float | None,
    use_subword: bool,
    token_remove_pattern: str | None,
) -> tuple[float | None, re.Pattern | None]:
    """
    Validate the parameters passed to the distillation function.

    :param vocabulary: The vocabulary to use.
    :param apply_zipf: DEPRECATED: This parameter used to control whether Zipf is applied.
        Zipf weighting is now controlled by the sif_coefficient parameter. If this is set to None, no weighting is applied.
    :param sif_coefficient: The SIF coefficient to use. If this is None, no weighting is applied.
        Should be a value >= 0 and < 1.0. A value of 1e-4 is a good default.
    :param use_subword: Whether to keep subword tokens in the vocabulary. If this is False, you must pass a vocabulary, and the returned tokenizer will only detect full words.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
    :return: The SIF coefficient to use.
    :raises: ValueError if the PCA dimension is larger than the number of dimensions in the embeddings.
    :raises: ValueError if the vocabulary contains duplicate tokens.
    :raises: ValueError if the regex can't be compiled.
    :raises: ValueError if the vocabulary is empty after token removal.

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

    if not use_subword and vocabulary is None:
        raise ValueError(
            "You must pass a vocabulary if you don't use subword tokens. Either pass a vocabulary, or set use_subword to True."
        )

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
    use_subword: bool = True,
    token_remove_pattern: str | None = r"\[unused\d+\]",
    trust_remote_code: bool = False,
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
    :param use_subword: Whether to keep subword tokens in the vocabulary. If this is False, you must pass a vocabulary, and the returned tokenizer will only detect full words.
    :param token_remove_pattern: If this is set to a string, we compile this into a regex. Any tokens that conform to this regex pattern will be removed from the vocabulary.
    :param trust_remote_code: Whether to trust the remote code. If this is False, we will only load components coming from `transformers`. If this is True, we will load all components.
    :return: A StaticModel

    """
    model: PreTrainedModel = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    return distill_from_model(
        model=model,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        device=device,
        pca_dims=pca_dims,
        apply_zipf=apply_zipf,
        use_subword=use_subword,
        token_remove_pattern=token_remove_pattern,
        sif_coefficient=sif_coefficient,
    )


def _post_process_embeddings(
    embeddings: np.ndarray, pca_dims: PCADimType, sif_coefficient: float | None = 1e-4
) -> np.ndarray:
    """Post process embeddings by applying PCA and SIF weighting by estimating the frequencies through Zipf's law."""
    if pca_dims is not None:
        if pca_dims == "auto":
            pca_dims = embeddings.shape[1]
        if pca_dims > embeddings.shape[1]:
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of dimensions in the embeddings ({embeddings.shape[1]}). "
                "Applying PCA, but not reducing dimensionality. Is this is not desired, please set `pca_dims` to None. "
                "Applying PCA will probably improve performance, so consider just leaving it."
            )
            pca_dims = embeddings.shape[1]
        if pca_dims >= embeddings.shape[0]:
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of tokens in the vocabulary ({embeddings.shape[0]}). Not applying PCA."
            )
        elif pca_dims <= embeddings.shape[1]:
            if isinstance(pca_dims, float):
                logger.info(f"Applying PCA with {pca_dims} explained variance.")
            else:
                logger.info(f"Applying PCA with n_components {pca_dims}")

            orig_dims = embeddings.shape[1]
            p = PCA(n_components=pca_dims, svd_solver="full")
            embeddings = p.fit_transform(embeddings)

            if embeddings.shape[1] < orig_dims:
                explained_variance_ratio = np.sum(p.explained_variance_ratio_)
                explained_variance = np.sum(p.explained_variance_)
                logger.info(f"Reduced dimensionality from {orig_dims} to {embeddings.shape[1]}.")
                logger.info(f"Explained variance ratio: {explained_variance_ratio:.3f}.")
                logger.info(f"Explained variance: {explained_variance:.3f}.")

    if sif_coefficient is not None:
        logger.info("Estimating word frequencies using Zipf's law, and then applying SIF.")
        inv_rank = 1 / (np.arange(2, embeddings.shape[0] + 2))
        proba = inv_rank / np.sum(inv_rank)
        embeddings *= (sif_coefficient / (sif_coefficient + proba))[:, None]

    return embeddings


def _clean_vocabulary(tokenizer: Tokenizer, vocabulary: list[str], added_tokens: list[str]) -> list[str]:
    """Cleans a vocabulary by removing duplicates and tokens that were already in the vocabulary."""
    added_tokens_set = set(added_tokens)
    seen_tokens = set()
    cleaned_vocabulary = []
    n_empty = 0
    n_duplicates = 0
    n_multiword = 0
    for token in vocabulary:
        if tokenizer.normalizer is not None:
            token = tokenizer.normalizer.normalize_str(token)

        if not token:
            n_empty += 1
            continue
        if token in seen_tokens or token in added_tokens_set:
            n_duplicates += 1
            continue

        seen_tokens.add(token)
        cleaned_vocabulary.append(token)

    if n_duplicates:
        logger.warning(f"Removed {n_duplicates} duplicate tokens.")
    if n_empty:
        logger.warning(f"Removed {n_empty} empty tokens.")
    if n_multiword:
        logger.warning(f"Removed {n_multiword} multiword tokens.")

    return cleaned_vocabulary
