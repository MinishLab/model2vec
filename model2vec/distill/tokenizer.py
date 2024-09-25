from __future__ import annotations

import json
import logging
from tempfile import NamedTemporaryFile

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def preprocess_vocabulary(tokenizer: Tokenizer, vocabulary: list[str]) -> list[str]:
    """Preprocess a vocabulary with a tokenizer by doing a roundtrip encode/decode."""
    encoded_ids: list[list[int]] = [
        encoding.ids for encoding in tokenizer.encode_batch(vocabulary, add_special_tokens=False)
    ]
    return tokenizer.decode_batch(encoded_ids)


def remove_tokens(tokenizer: Tokenizer, tokens_to_remove: list[str]) -> Tokenizer:
    """
    Remove tokens from a tokenizer.

    :param tokenizer: The tokenizer to remove tokens from.
    :param tokens_to_remove: The tokens to remove.
    :return: The modified tokenizer.
    """
    with NamedTemporaryFile(mode="w+", encoding="utf8") as temp_file:
        tokenizer.save(temp_file.name)
        data = json.load(temp_file)
        vocab: dict[str, int] = data["model"]["vocab"]

        n_tokens = len(vocab)
        for token in tokens_to_remove:
            if vocab.pop(token, None) is None:
                logger.warning(f"Token {token} was not in the vocabulary.")

        n_removed = n_tokens - len(vocab)
        logger.info(f"Removed {n_removed} tokens from the vocabulary.")

        reindexed = {token: idx for idx, (token, _) in enumerate(sorted(vocab.items(), key=lambda x: x[1]))}
        data["model"]["vocab"] = reindexed

        added_tokens = data["added_tokens"]
        added_tokens_to_keep = [token for token in added_tokens if token["content"] not in tokens_to_remove]
        data["added_tokens"] = added_tokens_to_keep
        tokenizer = Tokenizer.from_str(json.dumps(data))

    return tokenizer


def add_tokens(tokenizer: Tokenizer, tokens_to_add: list[str]) -> Tokenizer:
    """
    Add tokens to a tokenizer.

    :param tokenizer: The tokenizer to add tokens to.
    :param tokens_to_add: The tokens to add.
    :return: The modified tokenizer.
    """
    with NamedTemporaryFile(mode="w+") as temp_file:
        tokenizer.save(temp_file.name)
        data = json.load(open(temp_file.name))

        vocab: dict[str, int] = data["model"]["vocab"]
        for token in tokens_to_add:
            vocab[token] = len(vocab)

        tokenizer = Tokenizer.from_str(json.dumps(data))

    return tokenizer
