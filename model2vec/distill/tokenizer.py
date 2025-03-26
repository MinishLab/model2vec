from __future__ import annotations

import json
import logging
from typing import Any

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece

logger = logging.getLogger(__name__)


def _get_unk_token(tokenizer: Tokenizer) -> str | None:
    """
    Get the unknown token for a tokenizer.

    :param tokenizer: The tokenizer to extract the UNK token from
    :return: The unknown token string or None if not found
    """
    model = tokenizer.model
    if isinstance(model, WordPiece):
        return model.unk_token
    elif isinstance(model, BPE):
        return None
    elif isinstance(model, Unigram):
        unk_id: int | None = model.unk_id
        if unk_id is None:
            return None
        vocab = tokenizer.get_vocab()
        vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
        return vocab_sorted[unk_id]
    else:
        logger.warning(f"Unknown model type {type(model)}")
        return None


def _normalize_vocabulary(tokenizer: Tokenizer, vocabulary: list[str]) -> list[str]:
    """
    Normalize vocabulary tokens if a normalizer is present in the tokenizer.

    Only normalizes tokens that are not already in the tokenizer's vocabulary,
    to avoid normalizing tokens twice.

    :param tokenizer: The tokenizer to use.
    :param vocabulary: The vocabulary to normalize.
    :return: The normalized vocabulary.
    """
    current_tokenizer_vocab = set(tokenizer.get_vocab())
    normalized_tokens = []

    if tokenizer.normalizer is not None:
        for token in vocabulary:
            # Don't normalize twice, because normalization might not be idempotent.
            if token in current_tokenizer_vocab:
                normalized_tokens.append(token)
            else:
                normalized_tokens.append(tokenizer.normalizer.normalize_str(token))
    else:
        normalized_tokens = vocabulary

    return normalized_tokens


def _pre_tokenize_vocabulary(tokenizer: Tokenizer, tokens: list[str]) -> list[str]:
    """
    Apply pre-tokenization to vocabulary tokens if a pre-tokenizer is present.

    Only pre-tokenizes tokens that are not already in the tokenizer's vocabulary,
    to avoid processing tokens twice.

    :param tokenizer: The tokenizer to use.
    :param tokens: The tokens to pre-tokenize.
    :return: The pre-tokenized tokens.
    """
    current_tokenizer_vocab = set(tokenizer.get_vocab())
    pre_tokenized_tokens = []

    if tokenizer.pre_tokenizer is not None:
        for token in tokens:
            if token in current_tokenizer_vocab:
                pre_tokenized_tokens.append(token)
            else:
                # We know 100% sure that all pretokenized tokens will have length 1.
                pretokenized_tokens, _ = zip(*tokenizer.pre_tokenizer.pre_tokenize_str(f" {token}"))
                pre_tokenized_tokens.append(pretokenized_tokens[-1])
    else:
        pre_tokenized_tokens = tokens

    return pre_tokenized_tokens


def _make_new_merges_from_vocab(merges: list[tuple[str, str]], tokens: list[str]) -> list[tuple[str, str]]:
    """
    Generate new merges (bigrams) from a vocabulary.

    This function creates new merge pairs (bigrams) from a given vocabulary of tokens.
    The merges are used to build or extend a tokenizer's merge table.

    :param merges: The list of existing merges in the form "first second" where first and second are tokens.
    :param tokens: The list of tokens (vocabulary) from which to generate new merges.
    :return: The list of new merges in the form "first second" where first and second are tokens.
    """
    new_merges = merges.copy()
    current_vocab = set(tokens)
    already_merged = set("".join(merge) for merge in merges)

    for token in tokens:
        if token in already_merged:
            continue
        for needle in range(1, len(token)):
            first, second = token[:needle], token[needle:]
            if first in current_vocab and second in current_vocab:
                new_merges.append((first, second))

    return new_merges


def replace_vocabulary(tokenizer: Tokenizer, new_vocabulary: list[str]) -> Tokenizer:
    """Replace the vocabulary of a tokenizer with a new one."""
    tokenizer_json: dict[str, Any] = json.loads(tokenizer.to_str())

    # Use the new function in the replace_vocabulary function
    normalized_tokens = _normalize_vocabulary(tokenizer, new_vocabulary)
    # Very careful, we need to pretokenize words before adding them to the vocabulary.
    # But only if they are not subword tokens.
    pre_tokenized_tokens = _pre_tokenize_vocabulary(tokenizer, normalized_tokens)

    # Only keep UNK token

    # tokenizer_json.pop("special_tokens")
    model_type = tokenizer_json["model"]["type"]

    if model_type == "WordPiece":
        # Easiest, just add the new vocab
        unk_token = tokenizer_json["model"]["unk_token"]
        if unk_token is None:
            tokenizer_json["added_tokens"] = []
        else:
            tokenizer_json["added_tokens"] = [x for x in tokenizer_json["added_tokens"] if x["content"] == unk_token]

        tokenizer_json["model"]["vocab"] = {token: idx for idx, token in enumerate(pre_tokenized_tokens)}
    elif model_type == "Unigram":
        # Bit more difficult, we need to take into account probas.
        unk_id = tokenizer_json["model"]["unk_id"]
        if unk_id is None:
            tokenizer_json["added_tokens"] = []
        else:
            tokenizer_json["added_tokens"] = [x for x in tokenizer_json["added_tokens"] if x["id"] == unk_id]
        current_probas = dict(tokenizer_json["model"]["vocab"])
        lowest_proba = min(current_probas.values())
        new_probas = {word: current_probas.get(word, lowest_proba) for word in pre_tokenized_tokens}
        tokenizer_json["model"]["vocab"] = list(new_probas.items())
    elif model_type == "BPE":
        # Bit more difficult, we need to take into account merges.
        tokenizer_json["added_tokens"] = []
        tokenizer_json["model"]["vocab"] = {token: idx for idx, token in enumerate(pre_tokenized_tokens)}
        merges = tokenizer_json["model"]["merges"]
        merges = _make_new_merges_from_vocab(merges, pre_tokenized_tokens)
        tokenizer_json["model"]["merges"] = merges
    else:
        raise ValueError(f"Unknown model type {model_type}")

    return Tokenizer.from_str(json.dumps(tokenizer_json))
