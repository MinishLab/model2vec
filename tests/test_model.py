from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from model2vec import StaticModel


def test_initialization(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test successful initialization of StaticModel."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    assert model.embedding.weight.shape == (5, 2)
    assert len(model.tokens) == 5
    assert model.tokenizer == mock_tokenizer
    assert model.config == mock_config


def test_initialization_token_vector_mismatch(mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test if error is raised when number of tokens and vectors don't match."""
    mock_vectors = np.array([[0.1, 0.2], [0.2, 0.3]])
    with pytest.raises(ValueError):
        StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)


def test_encode_single_sentence(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of a single sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode("test sentence")
    assert encoded.shape == (2,)


def test_encode_multiple_sentences(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of multiple sentences."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode(["sentence 1", "sentence 2"])
    assert encoded.shape == (2, 2)


def test_encode_empty_sentence(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding with an empty sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode("")
    assert np.array_equal(encoded, np.zeros((2,)))


def test_normalize(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test normalization of vectors."""
    s = "word1 word2 word3"
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config, normalize=False)
    X = model.encode(s)
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config, normalize=True)
    normalized = model.encode(s)

    expected = X / np.linalg.norm(X)

    np.testing.assert_almost_equal(normalized, expected)


def test_save_pretrained(
    tmp_path: Path, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test saving a pretrained model."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)

    # Save the model to the tmp_path
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    # Check that the save_path directory contains the saved files
    assert save_path.exists()
    assert (save_path / "embeddings.safetensors").exists()
    assert (save_path / "tokenizer.json").exists()
    assert (save_path / "config.json").exists()


def test_load_pretrained(
    tmp_path: Path, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test loading a pretrained model after saving it."""
    # Save the model to a temporary path
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path)

    # Assert that the loaded model has the same properties as the original one
    np.testing.assert_array_equal(loaded_model.embedding.weight.numpy(), mock_vectors)
    assert loaded_model.tokenizer.get_vocab() == mock_tokenizer.get_vocab()
    assert loaded_model.config == mock_config


def test_initialize_normalize(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Tests whether the normalization initialization is correct."""
    model = StaticModel(mock_vectors, mock_tokenizer, {}, normalize=None)
    assert not model.normalize

    model = StaticModel(mock_vectors, mock_tokenizer, {}, normalize=False)
    assert not model.normalize

    model = StaticModel(mock_vectors, mock_tokenizer, {}, normalize=True)
    assert model.normalize

    model = StaticModel(mock_vectors, mock_tokenizer, {"normalize": False}, normalize=True)
    assert model.normalize

    model = StaticModel(mock_vectors, mock_tokenizer, {"normalize": True}, normalize=False)
    assert not model.normalize


def test_set_normalize(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Tests whether the normalize is set correctly."""
    model = StaticModel(mock_vectors, mock_tokenizer, {}, normalize=True)
    model.normalize = False
    assert model.config == {"normalize": False}
    model.normalize = True
    assert model.config == {"normalize": True}
