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
    # The normalized and pretokenized form of the token
    normalized_form: str
    # Whether the word is a continuing subword.
    is_subword: bool
    # Whether the token is internal to the model.
    is_internal: bool


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
