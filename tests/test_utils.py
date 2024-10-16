from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

from model2vec.distill.utils import select_optimal_device
from model2vec.hf_utils import _get_metadata_from_readme
from model2vec.utils import get_package_extras, importable


def test__get_metadata_from_readme_not_exists() -> None:
    """Test getting metadata from a README."""
    assert _get_metadata_from_readme(Path("zzz")) == {}


def test__get_metadata_from_readme_mocked_file() -> None:
    """Test getting metadata from a README."""
    with NamedTemporaryFile() as f:
        f.write(b"---\nkey: value\n---\n")
        f.flush()
        assert _get_metadata_from_readme(Path(f.name))["key"] == "value"


def test__get_metadata_from_readme_mocked_file_keys() -> None:
    """Test getting metadata from a README."""
    with NamedTemporaryFile() as f:
        f.write(b"")
        f.flush()
        assert set(_get_metadata_from_readme(Path(f.name))) == set()


@pytest.mark.parametrize(
    "device, expected, cuda, mps",
    [
        ("cpu", "cpu", True, True),
        ("cpu", "cpu", True, False),
        ("cpu", "cpu", False, True),
        ("cpu", "cpu", False, False),
        ("clown", "clown", False, False),
        (None, "cuda", True, True),
        (None, "cuda", True, False),
        (None, "mps", False, True),
        (None, "cpu", False, False),
    ],
)
def test_select_optimal_device(device: str | None, expected: str, cuda: bool, mps: bool) -> None:
    """Test whether the optimal device is selected."""
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
    ):
        assert select_optimal_device(device) == expected


def test_importable() -> None:
    """Test the importable function."""
    with pytest.raises(ImportError):
        importable("clown", "clown")

    importable("os", "clown")


def test_get_package_extras() -> None:
    """Test package extras."""
    extras = set(get_package_extras("model2vec", "distill"))
    assert extras == {"torch", "transformers", "typer", "scikit-learn"}


def test_get_package_extras_empty() -> None:
    """Test package extras with an empty package."""
    assert not list(get_package_extras("tqdm", ""))
