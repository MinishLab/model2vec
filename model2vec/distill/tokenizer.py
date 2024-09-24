from __future__ import annotations

import json
import logging
from tempfile import NamedTemporaryFile

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import Whitespace

logger = logging.getLogger(__name__)


def remove_tokens(tokenizer: Tokenizer, tokens_to_remove: list[str]) -> Tokenizer:
    """
    Remove tokens from a tokenizer.

    :param tokenizer: The tokenizer to remove tokens from.
    :param tokens_to_remove: The tokens to remove.
    :return: The modified tokenizer.
    """
    with NamedTemporaryFile(mode="w+") as temp_file:
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

        tokenizer = Tokenizer.from_str(json.dumps(data))

    return tokenizer


def create_tokenizer_from_vocab(vocabulary: list[str], unk_token: str, pad_token: str) -> Tokenizer:
    """
    Create a word level tokenizer from a vocabulary.

    :param vocabulary: The vocabulary mapping from strings to integers.
    :param unk_token: The string representing the unk token.
    :param pad_token: The string representing the pad token.
    :return: A word level tokenizer.
    :raises ValueError: If the unk_token or pad_token is not in the vocabulary.
    """
    vocabulary_indexed = {token: idx for idx, token in enumerate(vocabulary)}

    if unk_token not in vocabulary_indexed:
        raise ValueError(f"unk token: {unk_token} was not in your vocabulary.")
    if pad_token not in vocabulary_indexed:
        raise ValueError(f"pad token: {pad_token} was not in your vocabulary.")

    model = WordLevel(vocab=vocabulary_indexed, unk_token=unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()

    # NOTE: unk and pad token can be cased, doesn't matter.
    is_cased = any(token != token.lower() for token in vocabulary if token not in {unk_token, pad_token})

    normalization_steps = [NFKC()]
    if not is_cased:
        normalization_steps.append(Lowercase())
    tokenizer.normalizer = Sequence(normalization_steps)

    return tokenizer
