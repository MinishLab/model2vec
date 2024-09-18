from pathlib import Path

import numpy as np
import pytest
from transformers import PreTrainedTokenizerFast

from model2vec import StaticModel
from model2vec.distill.tokenizer import create_tokenizer_from_vocab


@pytest.fixture
def mock_tokenizer() -> PreTrainedTokenizerFast:
    """Create a mock tokenizer."""
    vocab = ["word1", "word2", "word3", "[UNK]", "[PAD]"]
    unk_token = "[UNK]"
    pad_token = "[PAD]"

    tokenizer = create_tokenizer_from_vocab(vocab, unk_token, pad_token)

    return tokenizer


@pytest.fixture
def mock_vectors() -> np.ndarray:
    """Create mock vectors."""
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def mock_config() -> dict[str, str]:
    """Create a mock config."""
    return {"some_config": "value"}


def test_initialization(
    mock_vectors: np.ndarray, mock_tokenizer: PreTrainedTokenizerFast, mock_config: dict[str, str]
) -> None:
    """Test successful initialization of StaticModel."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    assert model.vectors.shape == (5, 2)
    assert len(model.tokens) == 5
    assert model.tokenizer == mock_tokenizer
    assert model.config == mock_config


def test_initialization_token_vector_mismatch(
    mock_tokenizer: PreTrainedTokenizerFast, mock_config: dict[str, str]
) -> None:
    """Test if error is raised when number of tokens and vectors don't match."""
    mock_vectors = np.array([[0.1, 0.2], [0.2, 0.3]])  # Mismatch in size
    with pytest.raises(ValueError):
        StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)


def test_encode_single_sentence(
    mock_vectors: np.ndarray, mock_tokenizer: PreTrainedTokenizerFast, mock_config: dict[str, str]
) -> None:
    """Test encoding of a single sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode("test sentence")
    assert encoded.shape == (2,)  # 2-dimensional vector


def test_encode_multiple_sentences(
    mock_vectors: np.ndarray, mock_tokenizer: PreTrainedTokenizerFast, mock_config: dict[str, str]
) -> None:
    """Test encoding of multiple sentences."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode(["sentence 1", "sentence 2"])
    assert encoded.shape == (2, 2)  # Two 2-dimensional vectors


def test_encode_empty_sentence(
    mock_vectors: np.ndarray, mock_tokenizer: PreTrainedTokenizerFast, mock_config: dict[str, str]
) -> None:
    """Test encoding with an empty sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode("")
    assert np.array_equal(encoded, np.zeros((2,)))  # Should return a zero vector if sentence is empty


def test_normalize() -> None:
    """Test normalization of vectors."""
    X = np.array([[3, 4], [1, 2], [0, 0]])
    normalized = StaticModel.normalize(X)
    expected = np.array([[0.6, 0.8], [0.4472136, 0.89442719], [0, 0]])
    np.testing.assert_almost_equal(normalized, expected)


def test_save_pretrained(
    tmp_path: Path, mock_vectors: np.ndarray, mock_tokenizer: PreTrainedTokenizerFast, mock_config: dict[str, str]
) -> None:
    """Test saving a pretrained model using a tmp_path."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)

    # Save the model to the tmp_path
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    # Check that the save_path directory contains the saved files
    assert save_path.exists()
    assert (save_path / "embeddings.safetensors").exists()
    assert (save_path / "tokenizer.json").exists()
    assert (save_path / "tokenizer_config.json").exists()
    assert (save_path / "config.json").exists()
    assert (save_path / "special_tokens_map.json").exists()
