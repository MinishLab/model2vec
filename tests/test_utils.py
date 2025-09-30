from __future__ import annotations

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
    "torch_version, device, expected, cuda, mps, should_raise",
    [
        ("2.7.0", "cpu", "cpu", True, True, False),
        ("2.8.0", "cpu", "cpu", True, True, False),
        ("2.7.0", "clown", "clown", False, False, False),
        ("2.8.0", "clown", "clown", False, False, False),
        ("2.7.0", "mps", "mps", False, True, False),
        ("2.8.0", "mps", None, False, True, True),
        ("2.7.0", None, "cuda", True, True, False),
        ("2.7.0", None, "mps", False, True, False),
        ("2.7.0", None, "cpu", False, False, False),
        ("2.8.0", None, "cuda", True, True, False),
        ("2.8.0", None, "cpu", False, True, False),
        ("2.8.0", None, "cpu", False, False, False),
        ("2.9.0", None, "cpu", False, True, False),
        ("3.0.0", None, "cpu", False, True, False),
    ],
)
def test_select_optimal_device(torch_version, device, expected, cuda, mps, should_raise) -> None:
    """Test whether the optimal device is selected across versions and backends."""
    with (
        patch("torch.cuda.is_available", return_value=cuda),
        patch("torch.backends.mps.is_available", return_value=mps),
        patch("torch.__version__", torch_version),
    ):
        if should_raise:
            with pytest.raises(RuntimeError):
                select_optimal_device(device)
        else:
            assert select_optimal_device(device) == expected


def test_importable() -> None:
    """Test the importable function."""
    with pytest.raises(ImportError):
        importable("clown", "clown")

    importable("os", "clown")


def test_get_package_extras() -> None:
    """Test package extras."""
    extras = set(get_package_extras("model2vec", "distill"))
    assert extras == {"torch", "transformers", "scikit-learn"}


def test_get_package_extras_empty() -> None:
    """Test package extras with an empty package."""
    assert not list(get_package_extras("tqdm", ""))
