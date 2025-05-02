from __future__ import annotations

import re
from dataclasses import dataclass
from logging import getLogger

import torch

logger = getLogger(__name__)


@dataclass
class Token:
    """A class to represent a token."""

    form: str
    # Whether the word is a continuing subword.
    is_subword: bool
    # Whether it should be pretokenized.
    # This is independent of is_subword, because some
    # tokenizer models like BPE and Unigram do not have a
    # continuing subword prefix, but instead prefix nonsubwords.
    should_be_pretokenized: bool


def select_optimal_device(device: str | None) -> str:
    """
    Guess what your optimal device should be based on backend availability.

    If you pass a device, we just pass it through.

    :param device: The device to use. If this is not None you get back what you passed.
    :return: The selected device.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Automatically selected device: {device}")

    return device


def filter_vocabulary_by_regex(token_remove_regex: re.Pattern, tokens: list[tuple[str, int]]) -> list[int]:
    """
    Filter a sorted vocabulary by a regex pattern and return their ids.

    :param token_remove_regex: The regex pattern to filter by.
    :param tokens: The tokens to filter. This should be a list of tuples with the token and its id.
    :return: The ids of the tokens left after filtering.
    :raises ValueError: If no tokens are left after filtering.
    """
    id_list = []
    for token, id in tokens:
        if not token_remove_regex.match(token):
            id_list.append(id)

    if not id_list:
        raise ValueError("No tokens left after filtering by regex.")

    return id_list
