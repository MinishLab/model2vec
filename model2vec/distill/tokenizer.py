from __future__ import annotations

import json
import logging
from typing import Any

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


_DEFAULT_POST_PROCESSOR_TEMPLATE = {
    "type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}],
    "special_tokens": {},
}


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


def _remap_added_tokens(
    special_tokens: list[dict[str, Any]],
    vocabulary: list[str],
) -> list[dict[str, int]]:
    """
    Remap special tokens in the tokenizer.

    This function updates the special tokens in the tokenizer based on a mapping provided.
    It also ensures that the special tokens are present in the vocabulary.

    :param special_tokens: The special tokens to remap.
    :param vocabulary: The vocabulary as a list of tokens.
    :return: The updated special tokens.
    """
    # Deepcopy
    special_tokens = [{**x} for x in special_tokens]
    for token in special_tokens:
        token["id"] = vocabulary.index(token["content"])

    return special_tokens


def _make_new_merges_from_vocab(
    merges: list[tuple[str, str]], tokens: list[str], special_tokens: set[str | None]
) -> list[tuple[str, str]]:
    """
    Generate new merges from a vocabulary.

    This function creates new merge pairs from a given vocabulary of tokens.
    The merges are used to build or extend a tokenizer's merge table.

    :param merges: The list of existing merges in the form (first, second) where first and second are tokens.
    :param tokens: The list of tokens (vocabulary) from which to generate new merges.
    :param special_tokens: Tokens that should not be merged.
    :return: The list of new merges in the form (first, second) where first and second are tokens.
    """
    new_merges = merges.copy()
    current_vocab = set(tokens) - special_tokens
    already_merged = set("".join(merge) for merge in merges)

    for token in tokens:
        if token in special_tokens:
            continue
        if token in already_merged:
            continue
        if len(token) == 1:
            continue
        merges = []
        for index in range(1, len(token)):
            first, second = token[:index], token[index:]
            if first in current_vocab and second in current_vocab:
                merges.append((first, second))
        if not merges:
            logger.warning(f"Token {token} has no merges.")
            continue
        new_merges.extend(merges)

    return new_merges


def replace_vocabulary(
    tokenizer: Tokenizer, new_vocabulary: list[str], unk_token: str | None, pad_token: str | None
) -> Tokenizer:
    """Replace the vocabulary of a tokenizer with a new one."""
    tokenizer_json: dict[str, Any] = json.loads(tokenizer.to_str())

    # NOTE: all tokens have been normalized before.
    # Very careful, we need to pretokenize words before adding them to the vocabulary.
    # But only if they are not subword tokens.
    pre_tokenized_tokens = _pre_tokenize_vocabulary(tokenizer, new_vocabulary)

    model_type = tokenizer_json["model"]["type"]
    special_tokens = {unk_token, pad_token}

    if model_type in {"WordPiece", "BPE"}:
        # Easiest, just add the new vocab
        unk_token = unk_token or tokenizer_json["model"]["unk_token"]
        tokenizer_json["model"]["unk_token"] = unk_token
        tokenizer_json["added_tokens"] = [x for x in tokenizer_json["added_tokens"] if x["content"] in special_tokens]
        tokenizer_json["model"]["vocab"] = {token: idx for idx, token in enumerate(pre_tokenized_tokens)}

        if model_type == "BPE":
            # Bit more difficult, we need to take into account merges.
            merges = tokenizer_json["model"]["merges"]
            merges = _make_new_merges_from_vocab(merges, pre_tokenized_tokens, special_tokens)
            tokenizer_json["model"]["merges"] = merges

    elif model_type == "Unigram":
        # Bit more difficult, we need to take into account probas.
        unk_id = tokenizer_json["model"]["unk_id"]
        tokenizer_json["added_tokens"] = [x for x in tokenizer_json["added_tokens"] if x["content"] in special_tokens]
        vocab = tokenizer_json["model"]["vocab"]
        unk_token = vocab[unk_id][0] if unk_id is not None else None
        current_probas = dict(tokenizer_json["model"]["vocab"])
        lowest_proba = min(current_probas.values())
        new_probas = {word: current_probas.get(word, lowest_proba) for word in pre_tokenized_tokens}
        tokenizer_json["model"]["vocab"] = sorted(new_probas.items(), key=lambda x: x[1], reverse=True)

        tokens, _ = zip(*tokenizer_json["model"]["vocab"])
        tokenizer_json["model"]["unk_id"] = list(tokens).index(unk_token) if unk_token in tokens else None

    else:
        raise ValueError(f"Unknown model type {model_type}")

    # Remap special tokens
    added_tokens = tokenizer_json["added_tokens"]
    tokenizer_json["added_tokens"] = _remap_added_tokens(added_tokens, pre_tokenized_tokens)
    tokenizer_json["post_processor"] = _DEFAULT_POST_PROCESSOR_TEMPLATE

    return Tokenizer.from_str(json.dumps(tokenizer_json))
