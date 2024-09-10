# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Any, Protocol, cast

import click
import numpy as np
import safetensors
from rich.logging import RichHandler
from safetensors.numpy import save_file
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class SafeOpenProtocol(Protocol):
    """Protocol to fix safetensors safe open."""

    def get_tensor(self, key: str) -> np.ndarray:
        """Get a tensor."""
        ...


def setup_logging() -> None:
    """Simple logging setup."""
    logging.basicConfig(
        level="INFO",
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
    )


def save_pretrained(
    folder_path: Path, embeddings: np.ndarray, tokenizer: PreTrainedTokenizerFast, config: dict[str, Any]
) -> None:
    """
    Save a model to a folder.

    :param folder_path: The path to the folder.
    :param embeddings: The embeddings
    :param tokenizer: The tokenizer
    :param config: A metadata config
    """
    folder_path.mkdir(exist_ok=True, parents=True)
    save_file({"embeddings": embeddings}, folder_path / "embeddings.safetensors")
    tokenizer.save_pretrained(folder_path)
    json.dump(config, open(folder_path / "config.json", "w"))

    logger.info(f"Saved model to {folder_path}")


def load_pretrained(folder_path: Path) -> tuple[np.ndarray, PreTrainedTokenizerFast, dict[str, Any]]:
    """Loads a pretrained model from a folder."""
    opened_tensor_file = cast(
        SafeOpenProtocol, safetensors.safe_open(folder_path / "embeddings.safetensors", framework="numpy")
    )
    embeddings = opened_tensor_file.get_tensor("embeddings")

    tokenizer = AutoTokenizer.from_pretrained(folder_path)
    config = json.load(open(folder_path / "config.json"))

    if len(tokenizer.get_vocab()) != len(embeddings):
        logger.warning(
            f"Number of tokens does not match number of embeddings: `{len(tokenizer.get_vocab())}` vs `{len(embeddings)}`"
        )

    return embeddings, tokenizer, config
