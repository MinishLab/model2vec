# -*- coding: utf-8 -*-
import logging
from importlib import import_module
from importlib.metadata import metadata
from typing import Iterator

from rich.logging import RichHandler

logger = logging.getLogger(__name__)


_MODULE_MAP = (("scikit-learn", "sklearn"),)


def get_package_extras(package: str, extra: str) -> Iterator[str]:
    """Get the extras of the package."""
    message = metadata(package)
    all_packages = message.get_all("Requires-Dist")
    if all_packages is None:
        return
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
    logging.basicConfig(
        level="INFO",
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
