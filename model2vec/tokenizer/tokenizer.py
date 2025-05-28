from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, cast

from tokenizers import Tokenizer
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import (
    PreTokenizer,
)
from transformers import PreTrainedTokenizerFast

from model2vec.tokenizer.datamodels import Token
from model2vec.tokenizer.model import process_tokenizer
from model2vec.tokenizer.normalizer import replace_normalizer
from model2vec.tokenizer.pretokenizer import replace_pretokenizer

logger = logging.getLogger(__name__)


_DEFAULT_POST_PROCESSOR_TEMPLATE = {
    "type": "TemplateProcessing",
    "single": [{"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}],
    "special_tokens": {},
}


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


def replace_vocabulary(
    tokenizer: Tokenizer, new_vocabulary: list[Token], unk_token: str | None, pad_token: str | None
) -> Tokenizer:
    """Replace the vocabulary of a tokenizer with a new one."""
    tokenizer_json: dict[str, Any] = json.loads(tokenizer.to_str())
    added_tokens: list[dict[str, Any]] = tokenizer_json["added_tokens"]

    pre_tokenized_tokens = [x.normalized_form for x in new_vocabulary]

    # We need to remove the added tokens but keep [UNK] and [PAD] tokens.
    added_tokens = _rename_added_token(unk_token, "[UNK]", added_tokens, pre_tokenized_tokens)
    added_tokens = _rename_added_token(pad_token, "[PAD]", added_tokens, pre_tokenized_tokens)

    # Remove old added tokens from added tokens
    tokenizer_json["added_tokens"] = [x for x in added_tokens if x["content"] in {"[UNK]", "[PAD]"}]
    tokenizer_json = process_tokenizer(
        tokenizer_json, pre_tokenized_tokens, "[UNK]" if "[UNK]" in pre_tokenized_tokens else None
    )

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
    """Rename added tokens in the tokenizer."""
    if form is None:
        return added_tokens

    idx = vocabulary.index(form)
    added_token = [x for x in added_tokens if x["content"] == form]
    if added_token:
        added_token[0]["id"] = idx
        added_token[0]["content"] = new_form
        vocabulary[idx] = new_form

    return added_tokens


def clean_and_create_vocabulary(
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: list[str],
    token_remove_regex: re.Pattern | None,
) -> tuple[list[Token], Tokenizer]:
    """Cleans a vocabulary by removing duplicates and tokens that were already in the vocabulary."""
    seen_tokens = set()
    post_normalize_seen_tokens = set()
    n_empty = 0
    n_duplicates = 0

    backend_tokenizer = tokenizer.backend_tokenizer

    # Make a base list of tokens.
    internal_vocab: dict[str, int] = tokenizer.get_vocab()
    internal_tokens: list[str] = [k for k, _ in sorted(internal_vocab.items(), key=lambda x: x[1])]

    cleaned_vocabulary = _process_internal_tokens(tokenizer, backend_tokenizer, internal_tokens, token_remove_regex)
    # Copy the backend tokenizer to avoid modifying the original.
    backend_tokenizer = backend_tokenizer.from_str(backend_tokenizer.to_str())
    backend_tokenizer = replace_normalizer(backend_tokenizer)

    internal_tokens_set = {token.form for token in cleaned_vocabulary}

    normalizer: Normalizer | None = backend_tokenizer.normalizer
    for token in vocabulary:
        if normalizer is not None:
            token = cast(str, normalizer.normalize_str(token))

        if not token:
            n_empty += 1
            continue

        pre_tokenizer: PreTokenizer | None = backend_tokenizer.pre_tokenizer
        normalized_token = token
        if pre_tokenizer is not None:
            normalized_token = _normalize_vocabulary_token(
                token=token,
                pre_tokenizer=pre_tokenizer,
            )

        # We need to check whether the pretokenized token is in the vocabulary.
        # But we need to return the original token, because that will be tokenized
        # again by the tokenizer during featurization.
        if normalized_token in seen_tokens or normalized_token in internal_tokens_set:
            n_duplicates += 1
            continue

        # Add the possibly pretokenized token to seen
        seen_tokens.add(normalized_token)

        # After checking the token exists, we need to normalize it into the token
        # it will become. For byte tokens, this means we don't do anything. For
        # other types of tokens, we will insert a metaspace.
        # In the case of multiword tokens, we replace any spaces with the metaspace
        # or byte prefix token.
        if not normalized_token.startswith(("▁", "Ġ")):
            normalized_token = normalized_token.replace(" ", "▁")
            normalized_token = f"▁{normalized_token}"
        else:
            normalized_token = normalized_token.replace(" ", normalized_token[0])

        if normalized_token in post_normalize_seen_tokens:
            n_duplicates += 1
            continue

        post_normalize_seen_tokens.add(normalized_token)
        # Add the original string to the vocabulary.
        cleaned_vocabulary.append(
            Token(form=token, normalized_form=normalized_token, is_subword=False, is_internal=False)
        )

    if n_duplicates:
        logger.warning(f"Removed {n_duplicates} duplicate tokens.")
    if n_empty:
        logger.warning(f"Removed {n_empty} empty tokens.")

    return cleaned_vocabulary, replace_pretokenizer(backend_tokenizer)


def _process_internal_tokens(
    tokenizer: PreTrainedTokenizerFast,
    backend_tokenizer: Tokenizer,
    internal_tokens: list[str],
    token_remove_regex: re.Pattern | None,
) -> list[Token]:
    """Clean internal tokens."""
    # Get the pad and unk token from the tokenizer.
    pad_token: str | None = tokenizer.special_tokens_map.get("pad_token")  # type: ignore[assignment]
    unk_token: str | None = tokenizer.special_tokens_map.get("unk_token")  # type: ignore[assignment]
    # Empty set if no pad or unk token is set.
    added_tokens_to_keep: set[str] = {x for x in (pad_token, unk_token) if x is not None}
    added_tokens_to_remove = set(tokenizer.added_tokens_encoder) - added_tokens_to_keep
    cleaned_internal_tokens: list[Token] = []

    # Figure out whether token is a subword or not.
    encoded = backend_tokenizer.encode(f" {'a' * 25}", add_special_tokens=False)
    first_token, second_token, *_ = encoded.tokens
    # Isolate the prefix. We can't do first_token[0] because we don't know
    # how long the prefix is.
    # e.g., "Ġaaaa" -> "Ġ"
    a_index = None if "a" not in first_token else first_token.index("a")
    word_prefix = first_token[:a_index]
    is_byte_prefix = word_prefix == "Ġ"
    second_token = encoded.tokens[1]
    # The second token is the first subword token.
    # If a tokenizer uses subwords, this token will have been prefixed.
    # We don't know how long the prefix is.
    a_index = None if "a" not in second_token else second_token.index("a")
    subword_prefix = second_token[:a_index]

    pre_tokenizer: PreTokenizer | None = backend_tokenizer.pre_tokenizer

    for token in internal_tokens:
        # Create the token objects. If this returns None, it was unsucessful for some reason.
        if token_object := _create_single_internal_token(
            token=token,
            subword_prefix=subword_prefix,
            word_prefix=word_prefix,
            pre_tokenizer=pre_tokenizer,
            is_byte_prefix=is_byte_prefix,
            token_remove_regex=token_remove_regex,
            added_tokens_to_keep=added_tokens_to_keep,
            added_tokens_to_remove=added_tokens_to_remove,
        ):
            cleaned_internal_tokens.append(token_object)

    if len(cleaned_internal_tokens) != len(internal_tokens):
        logger.info(
            f"Removed {len(internal_tokens) - len(cleaned_internal_tokens)} internal tokens from the vocabulary."
        )

    return cleaned_internal_tokens


def _create_single_internal_token(
    token: str,
    subword_prefix: str,
    word_prefix: str,
    pre_tokenizer: PreTokenizer | None,
    is_byte_prefix: bool,
    token_remove_regex: re.Pattern | None,
    added_tokens_to_keep: set[str],
    added_tokens_to_remove: set[str],
) -> Token | None:
    """Create a token object from a string."""
    if token in added_tokens_to_remove:
        # We remove any tokens that are added tokens that aren't [UNK] or [PAD].
        return None
    if token in added_tokens_to_keep:
        # Don't put added tokens through the regular motions.
        return Token(form=token, normalized_form=token, is_subword=False, is_internal=True)
    if token_remove_regex and token_remove_regex.match(token):
        # If the regex matches, remove the token.
        return None

    # A token is a subword if there is a subword prefix and the word
    # starts with a subword prefix, or if there is a WORD prefix, and the word
    # does not start with this prefix. For metaspace tokenizers, for example:
    # "doghouse" -> ["_dog", "house"]
    # So we can only tell that "house" is a subword by knowing that it is not prefixed
    # and word-initial tokens are.
    is_subword = False
    if subword_prefix:
        is_subword = bool(token.startswith(subword_prefix))
    if word_prefix:
        is_subword = not bool(token.startswith(word_prefix))

    # Byte prefixed tokenizers don't need to be checked.
    if pre_tokenizer is not None and not is_byte_prefix:
        # We need to check the thing without prefixes. If we have a word prefix,
        # we need to check tokens that have are subwords. Other way around for subword
        # prefixes.
        if (subword_prefix and not is_subword) or (word_prefix and is_subword):
            # If this is True, the token is unreachable, even though it is a subword token.
            if len(pre_tokenizer.pre_tokenize_str(token)) > 1:
                return None

    # Turn a token into a normalized form for later processing.
    normalized_form = _create_normalized_form(token, subword_prefix, word_prefix, is_byte_prefix, is_subword)

    return Token(form=token, normalized_form=normalized_form, is_subword=is_subword, is_internal=True)


def _create_normalized_form(
    token: str, subword_prefix: str, word_prefix: str, is_byte_prefix: bool, is_subword: bool
) -> str:
    """Turn an internal token string into a normalized form."""
    # We don't need to check byte prefixed strings.
    if is_byte_prefix:
        return token
    # We need to check if the token is a subword or not and remove the prefix.
    if is_subword:
        return token.removeprefix(subword_prefix)
    # If the token is not a subword, we need to remove the word prefix, and add metaspace.
    return f"▁{token.removeprefix(word_prefix)}"


def turn_tokens_into_ids(
    tokens: list[Token], tokenizer: PreTrainedTokenizerFast, unk_token: str | None
) -> list[list[int]]:
    """
    Convert a list of Token objects to their corresponding token ID sequences.

    :param tokens: List of Token objects to convert
    :param tokenizer: The tokenizer to use for converting tokens to IDs
    :param unk_token: The string form of the unk token.
    :return: List of token IDs corresponding to the input tokens
    """
    unk_id = None if unk_token is None else tokenizer.convert_tokens_to_ids(unk_token)
    prefix, suffix = find_eos_bos(tokenizer)

    token_ids: list[list[int]] = []
    for token in tokens:
        if token.is_internal:
            # Careful. Any incorrect tokens will just get `[UNK]``, so this could go horribly wrong
            # Cast because return type is wrong.
            token_id: int = cast(int, tokenizer.convert_tokens_to_ids(token.form)) or 0
            # Explicitly check and warn if `unk_id` appears, but don't crash.
            if unk_id is not None and token_id == unk_id and token.form != unk_token:
                logger.warning(f"Token {token.form} was set to unk. This is wrong.")
            token_ids.append([*prefix, token_id, *suffix])
        else:
            token_ids.append(tokenizer.encode(token.form))

    return token_ids


def find_eos_bos(tokenizer: PreTrainedTokenizerFast) -> tuple[list[int], list[int]]:
    """Finds the eos and bos tokens for a tokenizer."""
    # Little bit complicated, because not all tokenizers have eos and bos tokens.
    encoding = tokenizer.encode("a", add_special_tokens=True)
    if len(encoding) != 3:
        a_encoded = tokenizer.encode("a", add_special_tokens=False)
        if len(a_encoded) != 1:
            raise ValueError(
                f"Error while encoding, couldn't determine eos and bos tokens. The model tokenizes 'a' to '{a_encoded}'"
            )
        a_idx = encoding.index(a_encoded[0])
        prefix, suffix = encoding[:a_idx], encoding[a_idx + 1 :]
    else:
        prefix, suffix = encoding[:1], encoding[2:]
    return prefix, suffix


def _normalize_vocabulary_token(token: str, pre_tokenizer: PreTokenizer) -> str:
    """Normalize a token that is not in the initial token vocabulary."""
    # Add prefix space for byte tokenizers.
    prefixed_token = f" {token}"
    pretokenized_tokens: tuple[str, ...]
    pretokenized_tokens, offsets = zip(*pre_tokenizer.pre_tokenize_str(prefixed_token))
    # The first item is always the start of the token.
    new_token = [pretokenized_tokens[0]]
    # Loop over the subtokens and offsets.
    for t, (s, _) in zip(pretokenized_tokens[1:], offsets[1:]):
        # Do not prefix the token with a space if it starts with a metaspace.
        if t.startswith("▁"):
            new_token.append(t)
        # If the character before the subtoken is a space, we have a
        # multiword token. e.g., "room for the moon", which is split into
        # ["room", "for", "the", "moon"].
        # If it doesn't have a space, it is part of a complex multiword token,
        # e.g., "chat-gpt", which is split into ["chat", "-", "gpt"].
        elif prefixed_token[s - 1] == " ":
            new_token.append(f" {t}")
        else:
            new_token.append(t)
    normalized_token = "".join(new_token)

    return normalized_token


def create_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    vocabulary: list[str],
    token_remove_regex: re.Pattern | None = None,
) -> PreTrainedTokenizerFast:
    """
    Create a tokenizer by adding tokens to the vocabulary.

    This function turns any tokenizer into a supertoken tokenizer. It does the following:
    1. Turns the tokenizer model into a unigram model.
    2. Adds a new pretokenizer, splitting on punctuation.
    3. Adds all tokens in vocabulary to the model.
    4. Removes any internal tokens that conform to the regex.

    :param tokenizer: The tokenizer to use.
    :param vocabulary: The vocabulary to use.
    :param token_remove_regex: The regex to use to remove tokens from the vocabulary.
    :return: The created tokenizer.
    """
    unk_token = cast(Optional[str], tokenizer.special_tokens_map.get("unk_token"))
    pad_token = cast(Optional[str], tokenizer.special_tokens_map.get("pad_token"))
    cleaned_vocabulary, backend_tokenizer = clean_and_create_vocabulary(tokenizer, vocabulary, token_remove_regex)
    new_tokenizer = replace_vocabulary(backend_tokenizer, cleaned_vocabulary, unk_token, pad_token)

    return PreTrainedTokenizerFast(tokenizer_object=new_tokenizer)
