from __future__ import annotations

import copy
import logging
import re
from typing import Sequence

import numpy as np
from skeletoken import TokenizerModel

from model2vec import StaticModel

logger = logging.getLogger(__name__)


def prune_vocabulary(model: StaticModel, vocabulary_to_prune: Sequence[str]) -> StaticModel:
    """Removes tokens from a model2vec model's vocabulary.

    :param model: The model2vec model to prune tokens from.
    :param vocabulary_to_prune: The tokens to remove from the vocabulary. Every token must already
        be present in the vocabulary.
    :return: A new model, with an updated embedding and vocabulary. The input model is left untouched.
    :raises ValueError: If the model is quantized, or if a token is not in the vocabulary.
    """
    if model.vocabulary_quantization is not None:
        raise ValueError("Cannot prune tokens from a quantized model.")

    tokenizer_model = TokenizerModel.from_tokenizer(model.tokenizer)
    tokenizer_model = tokenizer_model.remove_tokens_from_vocabulary(vocabulary_to_prune)

    delta = tokenizer_model.model_delta
    vocab_size = tokenizer_model.vocabulary_size
    embeddings = model.embedding

    new_embeddings = np.zeros((vocab_size, embeddings.shape[1]))
    if delta.token_mapping:
        to_ids, from_ids = zip(*delta.token_mapping.items())
        new_embeddings[np.asarray(to_ids)] = embeddings[np.asarray(from_ids)]

    new_weights = None
    if model.weights is not None:
        new_weights = np.ones(vocab_size, dtype=model.weights.dtype)
        if delta.token_mapping:
            to_ids, from_ids = zip(*delta.token_mapping.items())
            new_weights[np.asarray(to_ids)] = model.weights[np.asarray(from_ids)]

    return StaticModel(
        vectors=new_embeddings.astype(embeddings.dtype),
        tokenizer=tokenizer_model.to_tokenizer(),
        config=copy.deepcopy(model.config),
        normalize=model.normalize,
        base_model_name=model.base_model_name,
        language=copy.deepcopy(model.language),
        weights=new_weights,
        token_mapping=None,
        max_length=model.max_length,
    )


def add_vocabulary_to_model(model: StaticModel, vocabulary_to_add: Sequence[str]) -> StaticModel:
    """Adds tokens to a model2vec model.

    New tokens are initialized by encoding them with the existing model. Tokens that encode
    to a zero vector (e.g. because they contain no known subwords) are initialized randomly instead.

    :param model: The model2vec model to add tokens to.
    :param vocabulary_to_add: The vocabulary to add to the model.
    :return: A new model, with an updated embedding and vocabulary. The input model is left untouched.
    :raises ValueError: If the model is quantized.
    """
    if model.vocabulary_quantization is not None:
        raise ValueError("Cannot add tokens to a quantized model.")

    tokenizer_model = TokenizerModel.from_tokenizer(model.tokenizer)
    tokenizer_model = clean_and_create_vocabulary(tokenizer_model, vocabulary_to_add, None)

    delta = tokenizer_model.model_delta
    vocab_size = tokenizer_model.vocabulary_size
    embeddings = model.embedding

    new_embeddings = np.zeros((vocab_size, embeddings.shape[1]))
    if delta.token_mapping:
        to_ids, from_ids = zip(*delta.token_mapping.items())
        new_embeddings[np.asarray(to_ids)] = embeddings[np.asarray(from_ids)]

    new_weights = None
    if model.weights is not None:
        new_weights = np.ones(vocab_size, dtype=model.weights.dtype)
        if delta.token_mapping:
            to_ids, from_ids = zip(*delta.token_mapping.items())
            new_weights[np.asarray(to_ids)] = model.weights[np.asarray(from_ids)]

    if delta.new_tokens:
        new_tokens, new_ids = zip(*sorted(delta.new_tokens.items(), key=lambda x: x[1]))
        new_ids = np.asarray(new_ids)
        new_token_embeddings = model.encode(new_tokens).astype(new_embeddings.dtype)
        is_zero_vector = ~new_token_embeddings.any(axis=1)
        if is_zero_vector.any():
            rand_gen = np.random.default_rng()
            new_token_embeddings[is_zero_vector] = rand_gen.normal(
                size=(int(is_zero_vector.sum()), embeddings.shape[1])
            )
        new_embeddings[new_ids] = new_token_embeddings

    return StaticModel(
        vectors=new_embeddings.astype(embeddings.dtype),
        tokenizer=tokenizer_model.to_tokenizer(),
        config=copy.deepcopy(model.config),
        normalize=model.normalize,
        base_model_name=model.base_model_name,
        language=copy.deepcopy(model.language),
        weights=new_weights,
        token_mapping=None,
        max_length=model.max_length,
    )


def clean_and_create_vocabulary(
    model: TokenizerModel,
    vocabulary_to_add: Sequence[str],
    token_remove_regex: re.Pattern[str] | None,
) -> TokenizerModel:
    """Clean a vocabulary by removing duplicates and tokens that were already in the vocabulary.

    This function removes duplicate tokens and tokens that are already in the model's vocabulary.
    It also removes the tokenizer's post-processor, which we do not use anyway.

    :param model: The tokenizer model to clean.
    :param vocabulary_to_add: The vocabulary to add to the model. Any tokens in this vocabulary that
        are split according to the pretokenizer are added as multiword tokens.
    :param token_remove_regex: A regex pattern to remove tokens from the vocabulary.
    :return: The cleaned tokenizer model.
    """
    seen_tokens = set()

    n_duplicate = 0
    n_empty = 0
    n_regex_removed = 0

    # Remove the post processor.
    model.post_processor = None

    internal_tokens: list[str] = model.sorted_vocabulary
    if token_remove_regex:
        tokens_to_remove = [token for token in internal_tokens if token_remove_regex.match(token)]
        model = model.remove_tokens_from_vocabulary(tokens_to_remove)
        n_regex_removed = len(tokens_to_remove)
    preprocessor = model.preprocessor

    seen_tokens = set(internal_tokens)
    tokens_to_add: list[str] = []
    added_tokens_to_add: list[str] = []
    seen_added = set()
    for token in vocabulary_to_add:
        preprocessed = preprocessor.preprocess(token, had_initial_subword_prefix=True)
        if len(preprocessed) < 1:
            logger.warning(f"Token '{token}' was empty after preprocessing.")
            n_empty += 1
            continue
        if len(preprocessed) > 1:
            tokens_as_str = [f"'{subword}'" for subword in preprocessed]
            split_into = ",".join(tokens_as_str)
            logger.warning(
                f"Token '{token}' was split into multiple tokens after preprocessing: [{split_into}], adding it as a multi-word token."
            )
            if token in model.vocabulary:
                # If the unprocessed token (incorrectly) is in the vocabulary, we should remove it.
                model = model.remove_token_from_vocabulary(token)
            if preprocessor.normalizer:
                token = preprocessor.normalizer.normalize_str(token)
            # We need to strip because our AddedTokens also get stripped
            token = token.strip()
            if token in seen_added:
                logger.warning(f"Normalized added token '{token}' was in the added vocabulary twice.")
                continue
            added_tokens_to_add.append(token)
            seen_added.add(token)
            continue
        token = preprocessed[0]
        if token in seen_tokens:
            logger.warning(f"Token '{token}' was already in the vocabulary.")
            n_duplicate += 1
            continue
        if token_remove_regex and token_remove_regex.match(token):
            logger.warning(f"Token '{token}' was removed due to regex match.")
            n_regex_removed += 1
            continue
        seen_tokens.add(token)
        tokens_to_add.append(token)

    # Remove all prior added tokens
    model = model.prune_added_tokens()
    # Preprocess tokens is False because tokens are already preprocessed.
    model = model.add_tokens_to_vocabulary(tokens_to_add, preprocess_tokens=False)
    # The tokens are not special tokens, not single words, and already normalized.
    model = model.add_addedtokens(added_tokens_to_add, is_special=False, single_word=False, normalized=True)

    n_multiword = len(added_tokens_to_add)
    _report_statistics(n_multiword, n_duplicate, n_regex_removed, n_empty)

    return model


def _report_statistics(n_multiword: int, n_duplicate: int, n_regex_removed: int, n_empty: int) -> None:
    """Report statistics on the various types of issues we found."""
    if n_multiword:
        logger.info(f"Added {n_multiword} multi-word tokens to the vocabulary.")
    if n_duplicate:
        logger.info(f"Removed {n_duplicate} duplicate tokens.")
    if n_regex_removed:
        logger.info(f"Removed {n_regex_removed} tokens due to regex match.")
    if n_empty:
        logger.info(f"Removed {n_empty} empty tokens.")


def turn_tokens_into_ids(tokens: list[str], model: TokenizerModel) -> list[list[int]]:
    """Convert a list of Token objects to their corresponding token ID sequences.

    :param tokens: List of Token objects to convert
    :param model: The tokenizermodel of the tokenizer.
    :return: List of token IDs corresponding to the input tokens
    """
    prefix, suffix = model.bos_ids or [], model.eos_ids or []
    vocabulary = model.vocabulary
    tokenizer = model.to_tokenizer()

    token_ids: list[list[int]] = []
    for token in tokens:
        token_id = vocabulary.get(token)
        if token_id is not None:
            token_ids.append([*prefix, token_id, *suffix])
        else:
            token_ids.append(tokenizer.encode(token).ids)

    return token_ids
