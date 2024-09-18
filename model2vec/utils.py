# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Any, Protocol, cast

import click
import huggingface_hub
import numpy as np
import safetensors
from rich.logging import RichHandler
from safetensors.numpy import save_file
from tokenizers import Tokenizer

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


def save_pretrained(folder_path: Path, embeddings: np.ndarray, tokenizer: Tokenizer, config: dict[str, Any]) -> None:
    """
    Save a model to a folder.

    :param folder_path: The path to the folder.
    :param embeddings: The embeddings
    :param tokenizer: The tokenizer
    :param config: A metadata config
    """
    folder_path.mkdir(exist_ok=True, parents=True)
    save_file({"embeddings": embeddings}, folder_path / "embeddings.safetensors")
    tokenizer.save(str(folder_path / "tokenizer.json"))
    json.dump(config, open(folder_path / "config.json", "w"))

    logger.info(f"Saved model to {folder_path}")


def load_pretrained(
    folder_or_repo_path: str | Path, huggingface_token: str | None = None
) -> tuple[np.ndarray, Tokenizer, dict[str, Any]]:
    """
    Loads a pretrained model from a folder.

    :param folder_or_repo_path: The folder or repo path to load from.
        - If this is a local path, we will load from the local path.
        - If the local path is not found, we will attempt to load from the huggingface hub.
    :param huggingface_token: The huggingface token to use.
    :raises: FileNotFoundError if the folder exists, but the file does not exist locally.
    :return: The embeddings, tokenizer, and config.

    """
    folder_or_repo_path = Path(folder_or_repo_path)
    if folder_or_repo_path.exists():
        embeddings_path = folder_or_repo_path / "embeddings.safetensors"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file does not exist in {folder_or_repo_path}")

        config_path = folder_or_repo_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist in {folder_or_repo_path}")

        tokenizer_path = folder_or_repo_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file does not exist in {folder_or_repo_path}")

    else:
        logger.info("Folder does not exist locally, attempting to use huggingface hub.")
        embeddings_path = huggingface_hub.hf_hub_download(
            str(folder_or_repo_path), "embeddings.safetensors", token=huggingface_token
        )
        config_path = huggingface_hub.hf_hub_download(str(folder_or_repo_path), "config.json", token=huggingface_token)
        tokenizer_path = huggingface_hub.hf_hub_download(
            str(folder_or_repo_path), "tokenizer.json", token=huggingface_token
        )

    opened_tensor_file = cast(SafeOpenProtocol, safetensors.safe_open(embeddings_path, framework="numpy"))
    embeddings = opened_tensor_file.get_tensor("embeddings")

    tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
    config = json.load(open(config_path))

    if len(tokenizer.get_vocab()) != len(embeddings):
        logger.warning(
            f"Number of tokens does not match number of embeddings: `{len(tokenizer.get_vocab())}` vs `{len(embeddings)}`"
        )

    return embeddings, tokenizer, config
