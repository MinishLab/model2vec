from __future__ import annotations

import logging
import re
from typing import cast

from skeletoken import TokenizerModel
from skeletoken.addedtoken import AddedToken, AddedTokens
from skeletoken.models import WordPiece
from skeletoken.pretokenizers import ByteLevelPreTokenizer, PreTokenizerSequence
from tokenizers import Tokenizer
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import (
    PreTokenizer,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from model2vec.tokenizer.datamodels import Token

logger = logging.getLogger(__name__)


def _remap_added_tokens(
    added_tokens: list[AddedToken],
    vocabulary: list[str],
) -> list[AddedToken]:
    """
    Remap added tokens in the tokenizer.

    This function updates the added tokens in the tokenizer based on a mapping provided.
    It also ensures that the added tokens are present in the vocabulary.

    :param added_tokens: The added tokens to remap.
    :param vocabulary: The vocabulary as a list of tokens.
    :return: The updated added tokens.
    """
    for token in added_tokens:
        token.id = vocabulary.index(token.content)

    return added_tokens


def replace_vocabulary(tokenizer: Tokenizer, new_vocabulary: list[Token]) -> Tokenizer:
    """Replace the vocabulary of a tokenizer with a new one."""
    tokenizer_model = TokenizerModel.from_tokenizer(tokenizer)

    tokens = [token.normalized_form for token in new_vocabulary]

    assert isinstance(tokenizer_model.model, WordPiece)
    tokenizer_model.model.vocab.replace_vocabulary(tokens)

    new_added_tokens = []
    for added_token in tokenizer_model.added_tokens.root:
        if added_token.content not in {tokenizer_model.unk_token, tokenizer_model.pad_token}:
            continue
        new_added_tokens.append(added_token)
    for token in new_vocabulary:
        if token.is_multiword and token.form not in {tokenizer_model.unk_token, tokenizer_model.pad_token}:
            token_id = tokenizer_model.model.vocab[token.form]
            new_added_tokens.append(
                AddedToken(
                    content=token.form,
                    single_word=False,
                    lstrip=True,
                    rstrip=True,
                    normalized=True,
                    special=False,
                    id=token_id,
                )
            )

    pre_tokenized_tokens = [x.normalized_form for x in new_vocabulary]
    tokenizer_model.added_tokens = AddedTokens(_remap_added_tokens(new_added_tokens, pre_tokenized_tokens))
    # Set post processor to None because we don't care about it
    tokenizer_model.post_processor = None
    # We need to re-set the pad and unk tokens to put the correct indices.
    tokenizer_model.unk_token = tokenizer_model.unk_token
    tokenizer_model.pad_token = tokenizer_model.pad_token

    return tokenizer_model.to_tokenizer()


def clean_and_create_vocabulary(
    tokenizer: PreTrainedTokenizerFast,
    new_vocabulary: list[str],
    token_remove_regex: re.Pattern | None,
    lower_case: bool = True,
) -> tuple[list[Token], Tokenizer]:
    """Cleans a vocabulary by removing duplicates and tokens that were already in the vocabulary."""
    seen_tokens = set()
    n_empty = 0
    n_duplicates = 0

    tokenizer_model = _patch_tokenizer(tokenizer, lower_case=lower_case)

    # Make a base list of tokens.
    internal_vocab: dict[str, int] = tokenizer.get_vocab()
    internal_tokens: list[str] = [k for k, _ in sorted(internal_vocab.items(), key=lambda x: x[1])]

    # These need to be processed with the new backend tokenizer.
    cleaned_vocabulary = _process_internal_tokens(tokenizer_model, internal_tokens, token_remove_regex)
    internal_tokens_set = {token.form for token in cleaned_vocabulary}

    backend_tokenizer = tokenizer_model.to_tokenizer()
    normalizer: Normalizer | None = backend_tokenizer.normalizer
    for token in new_vocabulary:
        if normalizer is not None:
            token = cast(str, normalizer.normalize_str(token))

        if not token:
            n_empty += 1
            continue

        pre_tokenizer: PreTokenizer | None = backend_tokenizer.pre_tokenizer
        normalized_token = token
        is_multiword = False
        if pre_tokenizer is not None:
            tokens = pre_tokenizer.pre_tokenize_str(token)
            is_multiword = len(tokens) > 1
            if not tokens:
                # If no tokens are found, we have an empty string
                normalized_token = ""
            elif not is_multiword:
                # If it's not a multiword, take the first token.
                normalized_token = tokens[0][0]
            else:
                # This means it is a multiword.
                # Multiwords are found verbatim, so there's no need to change the normalized form.
                normalized_token = normalized_token

        # We need to check whether the pretokenized token is in the vocabulary.
        # But we need to return the original token, because that will be tokenized
        # again by the tokenizer during featurization.
        if token in seen_tokens or token in internal_tokens_set:
            n_duplicates += 1
            continue

        # Add the possibly pretokenized token to seen
        seen_tokens.add(token)
        # Add the original string to the vocabulary.
        cleaned_vocabulary.append(
            Token(
                form=token,
                normalized_form=normalized_token,
                is_subword=False,
                is_internal=False,
                is_multiword=is_multiword,
            )
        )

    if n_duplicates:
        logger.warning(f"Removed {n_duplicates} duplicate tokens.")
    if n_empty:
        logger.warning(f"Removed {n_empty} empty tokens.")

    return cleaned_vocabulary, backend_tokenizer


def _process_internal_tokens(
    tokenizer_model: TokenizerModel,
    internal_tokens: list[str],
    token_remove_regex: re.Pattern | None,
) -> list[Token]:
    """Clean internal tokens."""
    # Empty set if no pad or unk token is set.
    added_tokens_to_keep: set[str] = {
        x for x in (tokenizer_model.pad_token, tokenizer_model.unk_token) if x is not None
    }
    added_tokens_to_remove = {x.content for x in tokenizer_model.added_tokens.root} - added_tokens_to_keep
    cleaned_internal_tokens: list[Token] = []

    for token in internal_tokens:
        # Create the token objects. If this returns None, it was unsucessful for some reason.
        if token_object := _create_single_internal_token(
            token=token,
            subword_prefix=tokenizer_model.subword_prefix,
            word_prefix=tokenizer_model.word_prefix,
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
    subword_prefix: str | None,
    word_prefix: str | None,
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

    # For internal tokens, the normalized form is always the form.
    return Token(form=token, normalized_form=token, is_subword=is_subword, is_internal=True)


def turn_tokens_into_ids(tokens: list[Token], tokenizer: Tokenizer) -> list[list[int]]:
    """
    Convert a list of Token objects to their corresponding token ID sequences.

    :param tokens: List of Token objects to convert
    :param tokenizer: The tokenizer to use for converting tokens to IDs
    :return: List of token IDs corresponding to the input tokens
    """
    prefix, suffix = find_eos_bos(tokenizer)

    prefix_id, suffix_id = None, None
    vocab = tokenizer.get_vocab()
    if prefix is not None:
        prefix_id = vocab[prefix]
    if suffix is not None:
        suffix_id = vocab[suffix]

    token_ids: list[list[int]] = []
    for token in tokens:
        token_sequence = []
        if prefix_id is not None:
            token_sequence.append(prefix_id)
        if token.is_internal:
            token_id = vocab[token.form]
            token_sequence.append(token_id)
        else:
            token_sequence.extend(tokenizer.encode(token.form).ids)
        if suffix_id is not None:
            token_sequence.append(suffix_id)

        token_ids.append(token_sequence)

    return token_ids


def find_eos_bos(tokenizer: Tokenizer) -> tuple[str | None, str | None]:
    """Finds the eos and bos tokens for a tokenizer."""
    model = TokenizerModel.from_tokenizer(tokenizer)
    return model.bos, model.eos


def _patch_tokenizer(tokenizer: PreTrainedTokenizerFast, lower_case: bool = True) -> TokenizerModel:
    unk_token = cast(str | None, tokenizer.special_tokens_map.get("unk_token"))
    pad_token = cast(str | None, tokenizer.special_tokens_map.get("pad_token"))
    tokenizer_model = TokenizerModel.from_tokenizer(tokenizer.backend_tokenizer).make_model_greedy()

    # special_tokens map is preferred, then the original token is preferred, otherwise set manually.
    tokenizer_model.unk_token = unk_token or tokenizer_model.unk_token or "[UNK]"
    tokenizer_model.pad_token = pad_token or tokenizer_model.pad_token or "[PAD]"

    if lower_case:
        tokenizer_model = tokenizer_model.decase_vocabulary(remove_collisions=True)

    pretokenizer = tokenizer_model.pre_tokenizer
    if pretokenizer is not None:
        if isinstance(pretokenizer, PreTokenizerSequence):
            for pretokenizer_module in pretokenizer.pretokenizers:
                if isinstance(pretokenizer_module, ByteLevelPreTokenizer):
                    pretokenizer_module.add_prefix_space = True
        if isinstance(pretokenizer, ByteLevelPreTokenizer):
            pretokenizer.add_prefix_space = True

    return tokenizer_model
