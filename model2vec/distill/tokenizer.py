from __future__ import annotations

import json
import logging

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
    :raises ValueError: If the tokenizer model type is not supported.
    """
    model_vocab = set(tokenizer.get_vocab())
    # This triggers when tokens_to_remove is empty or when there is no overlap
    # between the tokens to remove and the model vocabulary.
    if not set(tokens_to_remove).intersection(model_vocab):
        # NOTE: return a copy.
        if tokens_to_remove:
            logger.info("No tokens to remove, none of the tokens were in the vocabulary.")
        else:
            logger.info("No tokens to remove.")
        return Tokenizer.from_str(tokenizer.to_str())

    tokenizer_data = json.loads(tokenizer.to_str())

    # Find all added tokens
    added_tokens = tokenizer_data["added_tokens"]
    added_tokens_str = {token["content"] for token in added_tokens}

    # Remove all added tokens from the list of tokens to remove.
    # Things will go bad if we keep them.
    tokens_to_remove = [token for token in tokens_to_remove if token not in added_tokens_str]

    # Load the vocabulary.
    model_type = tokenizer_data["model"]["type"]

    match model_type:
        case "WordPiece":
            # Vocab is a dictionary.
            vocab: dict[str, int] = tokenizer_data["model"]["vocab"]
            n_tokens = len(vocab)

            # Remove the tokens.
            for token in tokens_to_remove:
                if vocab.pop(token, None) is None:
                    logger.warning(f"Token {token} was not in the vocabulary.")

            n_removed = n_tokens - len(vocab)
            logger.info(f"Removed {n_removed} tokens from the vocabulary.")

            # Reindex the vocabulary so that it is contiguous.
            reindexed = {token: idx for idx, (token, _) in enumerate(sorted(vocab.items(), key=lambda x: x[1]))}
            tokenizer_data["model"]["vocab"] = reindexed
        case "Unigram":
            raise ValueError("Removing tokens from a unigram tokenizer is not supported.")
        case "BPE":
            raise ValueError("Removing tokens from a bpe tokenizer is not supported.")
        case _:
            raise ValueError(f"Unknown model type {model_type}")

    # Reindex the special tokens (i.e., CLS and SEP for BertTokenizers.)
    special_tokens_post_processor: dict[str, dict] = tokenizer_data["post_processor"]["special_tokens"]
    for token, token_data in special_tokens_post_processor.items():
        token_data["ids"] = [reindexed[token] for token in token_data["tokens"]]

    # Reinitialize the tokenizer from the json.
    tokenizer = Tokenizer.from_str(json.dumps(tokenizer_data))

    return tokenizer


def add_tokens(tokenizer: Tokenizer, tokens_to_add: list[str]) -> Tokenizer:
    """
    Add tokens to a tokenizer.

    :param tokenizer: The tokenizer to add tokens to.
    :param tokens_to_add: The tokens to add.
    :return: The modified tokenizer.
    :raises ValueError: If the tokenizer model type is not supported.
    """
    data = json.loads(tokenizer.to_str())

    model = data["model"]["type"]

    match model:
        case "WordPiece":
            wordpiece_vocab: dict[str, int] = data["model"]["vocab"]
            for token in tokens_to_add:
                if token not in wordpiece_vocab:
                    wordpiece_vocab[token] = len(wordpiece_vocab)
        case "Unigram":
            raise ValueError("Adding tokens to a unigram tokenizer is not supported.")
        case "BPE":
            raise ValueError("Adding tokens to a bpe tokenizer is not supported.")
        case _:
            raise ValueError(f"Unknown model type {model}")

    tokenizer = Tokenizer.from_str(json.dumps(data))

    return tokenizer
