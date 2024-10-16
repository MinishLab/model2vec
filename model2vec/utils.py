# -*- coding: utf-8 -*-
import json
import logging
from importlib import import_module
from importlib.metadata import metadata
from pathlib import Path
from typing import Iterator, Protocol, cast

import numpy as np
import safetensors
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class SafeOpenProtocol(Protocol):
    """Protocol to fix safetensors safe open."""

    def get_tensor(self, key: str) -> np.ndarray:
        """Get a tensor."""
        ...  # pragma: no cover


_MODULE_MAP = (("scikit-learn", "sklearn"),)


def get_package_extras(package: str, extra: str) -> Iterator[str]:
    """Get the extras of the package."""
    message = metadata(package)
    all_packages = message.get_all("Requires-Dist") or []
    for package in all_packages:
        name, *rest = package.split(";", maxsplit=1)
        if not rest:
            continue
        _, found_extra = rest[0].split("==", maxsplit=1)
        # Strip off quotes
        found_extra = found_extra.strip(' "')
        if found_extra == extra:
            yield name


def importable(module: str, extra: str) -> None:
    """Check if a module is importable."""
    module = dict(_MODULE_MAP).get(module, module)
    try:
        import_module(module)
    except ImportError:
        raise ImportError(
            f"`{module}`, is required. Please reinstall model2vec with the `distill` extra. `pip install model2vec[{extra}]`"
        )


def setup_logging() -> None:
    """Simple logging setup."""
    from rich.logging import RichHandler

    logging.basicConfig(
        level="INFO",
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def load_local_model(folder: Path) -> tuple[np.ndarray, Tokenizer, dict[str, str]]:
    """Load a local model."""
    embeddings_path = folder / "model.safetensors"
    tokenizer_path = folder / "tokenizer.json"
    config_path = folder / "config.json"

    opened_tensor_file = cast(SafeOpenProtocol, safetensors.safe_open(embeddings_path, framework="numpy"))
    embeddings = opened_tensor_file.get_tensor("embeddings")

    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = json.load(open(config_path))

    if len(tokenizer.get_vocab()) != len(embeddings):
        logger.warning(
            f"Number of tokens does not match number of embeddings: `{len(tokenizer.get_vocab())}` vs `{len(embeddings)}`"
        )

    return embeddings, tokenizer, config
