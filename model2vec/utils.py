from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from importlib import import_module
from importlib.metadata import metadata
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class SafeOpenProtocol(Protocol):
    """Protocol to fix safetensors safe open."""

    def get_tensor(self, key: str) -> np.ndarray:
        """Get a tensor."""
        ...  # pragma: no cover


_MODULE_MAP = (("scikit-learn", "sklearn"),)
_DIVIDERS = re.compile(r"[=<>!]+")


def get_package_extras(package: str, extra: str) -> Iterator[str]:
    """Get the extras of the package."""
    try:
        message = metadata(package)
    except Exception as e:
        raise ImportError(f"Could not retrieve metadata for package '{package}': {e}")

    all_packages = message.get_all("Requires-Dist") or []
    for package in all_packages:
        name, *rest = package.split(";", maxsplit=1)
        if rest:
            # Extract and clean the extra requirement
            found_extra = rest[0].split("==")[-1].strip(" \"'")
            if found_extra == extra:
                prefix, *_ = _DIVIDERS.split(name)
                yield prefix.strip()


def importable(module: str, extra: str) -> None:
    """Check if a module is importable."""
    module = dict(_MODULE_MAP).get(module, module)
    try:
        import_module(module)
    except ImportError:
        raise ImportError(
            f"`{module}`, is required. Please reinstall model2vec with the `{extra}` extra. `pip install model2vec[{extra}]`"
        )
