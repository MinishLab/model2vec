from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import safetensors
import safetensors.numpy
from tokenizers import Tokenizer

from model2vec.distill.utils import select_optimal_device
from model2vec.hf_utils import _get_metadata_from_readme
from model2vec.utils import get_package_extras, importable, load_local_model


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
    assert extras == {"torch", "transformers", "scikit-learn"}


def test_get_package_extras_empty() -> None:
    """Test package extras with an empty package."""
    assert not list(get_package_extras("tqdm", ""))


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"dog": "cat"}, {"dog": "cat"}),
        ({}, {}),
        (None, {}),
    ],
)
def test_local_load(mock_tokenizer: Tokenizer, config: dict[str, Any], expected: dict[str, Any]) -> None:
    """Test local loading."""
    x = np.ones((mock_tokenizer.get_vocab_size(), 2))

    with TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        safetensors.numpy.save_file({"embeddings": x}, Path(tempdir) / "model.safetensors")
        mock_tokenizer.save(str(Path(tempdir) / "tokenizer.json"))
        if config is not None:
            json.dump(config, open(tempdir_path / "config.json", "w"))
        arr, tokenizer, config = load_local_model(tempdir_path)
        assert config == expected
        assert tokenizer.to_str() == mock_tokenizer.to_str()
        assert arr.shape == x.shape


def test_local_load_mismatch(mock_tokenizer: Tokenizer, caplog: pytest.LogCaptureFixture) -> None:
    """Test local loading."""
    x = np.ones((10, 2))

    with TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        safetensors.numpy.save_file({"embeddings": x}, Path(tempdir) / "model.safetensors")
        mock_tokenizer.save(str(Path(tempdir) / "tokenizer.json"))

        load_local_model(tempdir_path)
        expected = (
            f"Number of tokens does not match number of embeddings: `{len(mock_tokenizer.get_vocab())}` vs `{len(x)}`"
        )
        assert len(caplog.records) == 1
        assert caplog.records[0].message == expected
