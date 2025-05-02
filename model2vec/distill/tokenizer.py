from __future__ import annotations

import json
import logging
from string import punctuation
from typing import Any

from tokenizers import Regex, Tokenizer
from tokenizers.normalizers import Lowercase, Normalizer, Replace, Strip
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import (
    BertPreTokenizer,
    ByteLevel,
    CharDelimiterSplit,
    Metaspace,
    PreTokenizer,
    Punctuation,
    Sequence,
    Split,
    UnicodeScripts,
    Whitespace,
    WhitespaceSplit,
)

_FORBIDDEN_PRETOKENIZERS = (
    BertPreTokenizer,
    CharDelimiterSplit,
    Metaspace,
    Punctuation,
    Split,
    UnicodeScripts,
    Whitespace,
    WhitespaceSplit,
)


from model2vec.distill.utils import Token

logger = logging.getLogger(__name__)


_DEFAULT_POST_PROCESSOR_TEMPLATE = {
    "type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}],
    "special_tokens": {},
}


def _pre_tokenize_vocabulary(tokenizer: Tokenizer, tokens: list[Token], subword_prefix: str) -> list[str]:
    """
    Apply pre-tokenization to vocabulary tokens if a pre-tokenizer is present.

    Only pre-tokenizes tokens that are not already in the tokenizer's vocabulary,
    to avoid processing tokens twice.

    :param tokenizer: The tokenizer to use.
    :param tokens: The tokens to pre-tokenize.
    :param subword_prefix: The prefix for subwords.
    :return: The pre-tokenized tokens.
    """
    pre_tokenized_tokens = []

    if tokenizer.pre_tokenizer is not None:
        for token in tokens:
            if token.is_subword:
                # Original tokens do not need to be pre-tokenized.
                form = token.form
                if subword_prefix is not None:
                    form = token.form.removeprefix(subword_prefix)
                pre_tokenized_tokens.append(form)
            elif token.should_be_pretokenized:
                # Join tokens just to be sure.
                token.form = tokenizer.normalizer.normalize_str(token.form).rstrip()
                pretokenized_tokens, _ = zip(*tokenizer.pre_tokenizer.pre_tokenize_str(token.form))
                form = " ".join(pretokenized_tokens)
                pre_tokenized_tokens.append(form)
            else:
                token.form = tokenizer.normalizer.normalize_str(token.form).rstrip()
                pre_tokenized_tokens.append(token.form)
    else:
        pre_tokenized_tokens = [token.form for token in tokens]

    return pre_tokenized_tokens


def _remap_added_tokens(
    special_tokens: list[dict[str, Any]],
    vocabulary: list[str],
) -> list[dict[str, Any]]:
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


def _prepare_normalizer(
    normalizer: Normalizer,
) -> Normalizer:
    """
    Prepare the normalizer for the tokenizer.

    This function sets the normalizer for the tokenizer based on the provided normalizer type.
    If no normalizer is provided, it uses the default one.

    :param normalizer: The tokenizer to prepare.
    :return: The prepared tokenizer.
    """
    new_normalizers = []
    for char in punctuation:
        new_normalizers.append(Replace(char, f" {char} "))
    new_normalizers.append(Replace(Regex(r"\s+"), " "))
    new_normalizers.append(Strip(right=True))
    if normalizer is None:
        return NormalizerSequence(new_normalizers)

    return NormalizerSequence([normalizer] + new_normalizers)


def _fix_single_pretokenizer(pretokenizer: PreTokenizer) -> PreTokenizer | None:
    """Fixes a single pretokenizer to allow multiword units."""
    if isinstance(pretokenizer, Metaspace):
        return Metaspace(split=False, replacement=pretokenizer.replacement, prepend_scheme=pretokenizer.prepend_scheme)
    if isinstance(pretokenizer, _FORBIDDEN_PRETOKENIZERS):
        return Metaspace(split=False, replacement="▁")
    elif isinstance(pretokenizer, ByteLevel):
        pretokenizer.use_regex = False
        pretokenizer.add_prefix_space = True

    return pretokenizer


def _fix_pretokenizer_for_super(pre: PreTokenizer | None) -> Tokenizer:
    """Fixes the pretokenizer to allow multiword units."""
    if pre is None:
        return pre

    if isinstance(pre, Sequence):
        return Metaspace(split=False)

    return _fix_single_pretokenizer(pre)


def _process_wordpiece(
    tokenizer_json: dict[str, Any], pre_tokenized_tokens: list[str], unk_token: str | None
) -> dict[str, Any]:
    """Process the WordPiece tokenizer JSON."""
    tokenizer_json["model"]["type"] = "Unigram"
    tokenizer_json["model"]["unk_id"] = pre_tokenized_tokens.index(unk_token) if unk_token else None
    tokenizer_json["model"]["vocab"] = [(token, 0.0) for token in pre_tokenized_tokens]

    return tokenizer_json


def _process_bpe(
    tokenizer_json: dict[str, Any], pre_tokenized_tokens: list[str], unk_token: str | None
) -> dict[str, Any]:
    """Process the BPE tokenizer JSON."""
    tokenizer_json["model"]["type"] = "Unigram"
    tokenizer_json["model"]["unk_id"] = pre_tokenized_tokens.index(unk_token) if unk_token else None
    tokenizer_json["model"]["vocab"] = [(token, 0.0) for token in pre_tokenized_tokens]

    return tokenizer_json


def _process_unigram(tokenizer_json: dict[str, Any], pre_tokenized_tokens: list[str], unk_token: str) -> dict[str, Any]:
    """Process the Unigram tokenizer JSON."""
    current_probas = dict(tokenizer_json["model"]["vocab"])
    avg_proba = sum(current_probas.values()) / len(current_probas)
    new_probas = {word: current_probas.get(word, avg_proba) for word in pre_tokenized_tokens}
    tokenizer_json["model"]["vocab"] = sorted(new_probas.items(), key=lambda x: x[1], reverse=True)

    tokens, _ = zip(*tokenizer_json["model"]["vocab"])
    tokenizer_json["model"]["unk_id"] = list(tokens).index(unk_token)

    return tokenizer_json


def replace_vocabulary(
    tokenizer: Tokenizer, new_vocabulary: list[Token], unk_token: str | None, pad_token: str | None
) -> Tokenizer:
    """Replace the vocabulary of a tokenizer with a new one."""
    tokenizer.normalizer = _prepare_normalizer(tokenizer.normalizer)
    tokenizer.pre_tokenizer = _fix_pretokenizer_for_super(tokenizer.pre_tokenizer)
    tokenizer_json: dict[str, Any] = json.loads(tokenizer.to_str())

    # NOTE: all tokens have been normalized before.
    # Very careful, we need to pretokenize words before adding them to the vocabulary.
    # But only if they are not part of the original vocabulary.
    subword_prefix = tokenizer_json["model"].get("continuing_subword_prefix", "")

    pre_tokenized_tokens = _pre_tokenize_vocabulary(tokenizer, new_vocabulary, subword_prefix=subword_prefix)

    model_type = tokenizer_json["model"]["type"]
    added_tokens: list[dict[str, Any]] = tokenizer_json["added_tokens"]

    # We need to remove the added tokens but keep [UNK] and [PAD] tokens.
    added_tokens = _rename_added_token(unk_token, "[UNK]", added_tokens, pre_tokenized_tokens)
    added_tokens = _rename_added_token(pad_token, "[PAD]", added_tokens, pre_tokenized_tokens)

    # Remove old added tokens from added tokens
    tokenizer_json["added_tokens"] = [x for x in added_tokens if x["content"] in {"[UNK]", "[PAD]"}]

    if model_type == "WordPiece":
        tokenizer_json = _process_wordpiece(tokenizer_json, pre_tokenized_tokens, "[UNK]")
    elif model_type == "BPE":
        tokenizer_json = _process_bpe(tokenizer_json, pre_tokenized_tokens, "[UNK]")
    elif model_type == "Unigram":
        tokenizer_json = _process_unigram(tokenizer_json, pre_tokenized_tokens, "[UNK]")
    else:
        raise ValueError(f"Unknown model type {model_type}")

    # Remap special tokens
    tokenizer_json["added_tokens"] = _remap_added_tokens(
        special_tokens=tokenizer_json["added_tokens"],
        vocabulary=pre_tokenized_tokens,
    )
    tokenizer_json["post_processor"] = _DEFAULT_POST_PROCESSOR_TEMPLATE

    return Tokenizer.from_str(json.dumps(tokenizer_json))


def _rename_added_token(
    form: str | None, new_form: str, added_tokens: list[dict[str, Any]], vocabulary: list[str]
) -> list[dict[str, Any]]:
    """Rename special tokens in the tokenizer."""
    if form is None:
        return added_tokens

    idx = vocabulary.index(form)
    added_token = [x for x in added_tokens if x["content"] == form]
    if added_token:
        added_token[0]["id"] = idx
        added_token[0]["content"] = new_form
        vocabulary[idx] = new_form

    return added_tokens
