import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from tokenizers import Tokenizer

from model2vec.model import StaticModel
from model2vec.persistence.hf import maybe_get_cached_model_path


def test_local_loading(mock_static_model: StaticModel) -> None:
    """Test saving and loading."""
    with TemporaryDirectory() as dir_name:
        mock_static_model.save_pretrained(dir_name)

        with patch(
            "model2vec.persistence.persistence.huggingface_hub.snapshot_download",
        ) as mock_snapshot:
            mock_snapshot.return_value = Path(dir_name)
            # Patch the cache to return the local path, simulate a cache hit
            # we pass a fake model name to actually hit the cache and snapshot download paths
            with patch("model2vec.persistence.persistence.maybe_get_cached_model_path") as cache:
                # Simulate cache hit
                cache.return_value = Path(dir_name)
                s = StaticModel.from_pretrained("my_org/haha", force_download=True)
                assert mock_snapshot.call_args[0] == ("my_org/haha",)
                assert s.tokens == mock_static_model.tokens
                s = StaticModel.from_pretrained("my_org/haha", force_download=False)
                assert s.tokens == mock_static_model.tokens

                # Simulate cache miss
                cache.return_value = None
                s = StaticModel.from_pretrained("my_org/haha", force_download=True)
                assert s.tokens == mock_static_model.tokens
                s = StaticModel.from_pretrained("my_org/haha", force_download=False)
                assert s.tokens == mock_static_model.tokens

                # Called twice, only when `force_download` is False
                assert cache.call_count == 2
            # Called three times, two times when force download is True,
            # and once when it was false but the cache returned None
            assert mock_snapshot.call_count == 3


def test_garbage(mock_static_model: StaticModel) -> None:
    """Test that garbage loading crashes."""
    with TemporaryDirectory() as dir_name:
        mock_static_model.save_pretrained(dir_name)
        dir_name_path = Path(dir_name)
        shutil.move(dir_name_path / "model.safetensors", dir_name_path / "model.safetenso")
        with pytest.raises(ValueError):
            StaticModel.from_pretrained(dir_name)


def test_subfolder_loading(mock_static_model: StaticModel) -> None:
    """Test that subfolder loading works."""
    with TemporaryDirectory() as dir_name:
        dir_name_path = Path(dir_name)
        mock_static_model.save_pretrained(dir_name_path / "subfolder")
        with pytest.raises(ValueError):
            StaticModel.from_pretrained(dir_name)
        model = StaticModel.from_pretrained(dir_name, subfolder="subfolder")
        assert isinstance(model, StaticModel)


def test_save_pretrained_non_contiguous_embeddings(tmp_path: Path, mock_tokenizer: Tokenizer) -> None:
    """Non-contiguous embeddings (e.g. Fortran-ordered) must round-trip exactly through safetensors."""
    vectors = np.asfortranarray(np.random.RandomState(0).randn(len(mock_tokenizer.get_vocab()), 8))
    assert not vectors.flags["C_CONTIGUOUS"]

    model = StaticModel(vectors=vectors, tokenizer=mock_tokenizer)
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    loaded_model = StaticModel.from_pretrained(save_path)
    np.testing.assert_array_equal(loaded_model.embedding, vectors)


def test_save_pretrained_with_weights_and_mapping(tmp_path: Path, mock_tokenizer: Tokenizer) -> None:
    """save_pretrained must persist weights and a token mapping when present."""
    n_tokens = len(mock_tokenizer.get_vocab())
    vectors = np.random.RandomState(0).randn(3, 4)
    weights = np.random.RandomState(1).randn(n_tokens)
    mapping = np.array([0, 1, 2, 0, 1][:n_tokens])

    model = StaticModel(vectors=vectors, tokenizer=mock_tokenizer, weights=weights, token_mapping=mapping)
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    loaded_model = StaticModel.from_pretrained(save_path)
    np.testing.assert_array_equal(loaded_model.embedding, vectors)
    np.testing.assert_array_equal(loaded_model.weights, weights)
    np.testing.assert_array_equal(loaded_model.token_mapping, mapping)


def test_maybe_get_cached_model_path() -> None:
    """Test cached model path."""
    model_id = "t/t"
    with TemporaryDirectory() as temp_dir:
        with patch("model2vec.persistence.hf.HF_HUB_CACHE", temp_dir):
            # No slash
            assert maybe_get_cached_model_path("t") is None
            # More than 1 slash
            assert maybe_get_cached_model_path("t/t/t") is None
            # Not created yet
            assert maybe_get_cached_model_path(model_id) is None
            normalized = model_id.replace("/", "--")
            repo_dir = Path(temp_dir) / f"models--{normalized}"
            repo_dir.mkdir(parents=True)
            # Snapshot not created
            assert maybe_get_cached_model_path(model_id) is None
            with_snapshot = repo_dir / "snapshots"
            with_snapshot.mkdir(parents=True, exist_ok=True)
            # No snapshots yet
            assert maybe_get_cached_model_path(model_id) == None
            repo_dir_a = with_snapshot / "a"
            repo_dir_a.mkdir(parents=True, exist_ok=True)
            assert maybe_get_cached_model_path(model_id) == repo_dir_a
