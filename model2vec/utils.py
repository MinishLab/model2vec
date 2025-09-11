# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import re
from importlib import import_module
from importlib.metadata import metadata
from typing import Any, Iterator, Protocol, cast

import numpy as np
from joblib import Parallel
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressParallel(Parallel):
    """A drop-in replacement for joblib.Parallel that shows a tqdm progress bar."""

    def __init__(self, use_tqdm: bool = True, total: int | None = None, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the ProgressParallel object.

        :param use_tqdm: Whether to show the progress bar.
        :param total: Total number of tasks (batches) you expect to process. If None,
                    it updates the total dynamically to the number of dispatched tasks.
        :param *args: Additional arguments to pass to `Parallel.__init__`.
        :param **kwargs: Additional keyword arguments to pass to `Parallel.__init__`.
        """
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Create a tqdm context."""
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            self._pbar = self._pbar
            return super().__call__(*args, **kwargs)

    def print_progress(self) -> None:
        """Hook called by joblib as tasks complete. We update the tqdm bar here."""
        if self._total is None:
            # If no fixed total was given, we dynamically set the total
            self._pbar.total = self.n_dispatched_tasks
        # Move the bar to the number of completed tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


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
                prefix = prefix.split("@")[0].strip()
                yield prefix.strip()


def importable(module: str, extra: str) -> None:
    """Check if a module is importable."""
    module = dict(_MODULE_MAP).get(module, module)
    # Allows this to work with git installed modules.
    module = module.split("@")[0].strip()
    try:
        import_module(module)
    except ImportError:
        raise ImportError(
            f"`{module}`, is required. Please reinstall model2vec with the `{extra}` extra. `pip install model2vec[{extra}]`"
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
