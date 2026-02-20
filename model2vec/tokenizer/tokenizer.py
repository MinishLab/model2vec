from __future__ import annotations

import logging
import re

from skeletoken import TokenizerModel

logger = logging.getLogger(__name__)


def clean_and_create_vocabulary(
    model: TokenizerModel,
    vocabulary_to_add: list[str],
    token_remove_regex: re.Pattern[str] | None,
) -> TokenizerModel:
    """Clean a vocabulary by removing duplicates and tokens that were already in the vocabulary."""
    seen_tokens = set()

    n_duplicate = 0
    n_empty = 0
    n_regex_removed = 0

    internal_tokens: list[str] = model.sorted_vocabulary
    if token_remove_regex:
        tokens_to_remove = [token for token in internal_tokens if token_remove_regex.match(token)]
        model = model.remove_tokens_from_vocabulary(tokens_to_remove)
        n_regex_removed = len(tokens_to_remove)
    preprocessor = model.preprocessor

    seen_tokens = set(internal_tokens)
    tokens_to_add: list[str] = []
    added_tokens_to_add: list[str] = []
    for token in vocabulary_to_add:
        preprocessed = preprocessor.preprocess(token)
        if len(preprocessed) < 1:
            logger.warning(f"Token '{token}' was empty after preprocessing.")
            n_empty += 1
            continue
        if len(preprocessed) > 1:
            tokens_as_str = [f"'{subword}'" for subword in preprocessed]
            split_into = ",".join(tokens_as_str)
            logger.warning(f"Token '{token}' was split into multiple tokens after preprocessing: [{split_into}]")
            added_tokens_to_add.append(token)
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

    model = model.add_tokens_to_vocabulary(tokens_to_add, preprocess_tokens=True)
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
    """
    Convert a list of Token objects to their corresponding token ID sequences.

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
