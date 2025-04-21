from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import safetensors
from tokenizers import Tokenizer

from model2vec import StaticModel


def test_initialization(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test successful initialization of StaticModel."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    assert model.embedding.shape == (5, 2)
    assert len(model.tokens) == 5
    assert model.tokenizer == mock_tokenizer
    assert model.config == mock_config


def test_initialization_token_vector_mismatch(mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test if error is raised when number of tokens and vectors don't match."""
    mock_vectors = np.array([[0.1, 0.2], [0.2, 0.3]])
    with pytest.raises(ValueError):
        StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)


def test_tokenize(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test tokenization of a sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    model._can_encode_fast = True
    tokens_fast = model.tokenize(["word1 word2"])
    model._can_encode_fast = False
    tokens_slow = model.tokenize(["word1 word2"])

    assert tokens_fast == tokens_slow


def test_encode_batch_fast(
    mock_vectors: np.ndarray, mock_berttokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test tokenization of a sentence."""
    if hasattr(mock_berttokenizer, "encode_batch_fast"):
        del mock_berttokenizer.encode_batch_fast
        model = StaticModel(vectors=mock_vectors, tokenizer=mock_berttokenizer, config=mock_config)
        assert not model._can_encode_fast


def test_encode_single_sentence(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of a single sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode("word1 word2")
    assert encoded.shape == (2,)


def test_encode_single_sentence_empty(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of a single empty sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    model.normalize = True
    encoded = model.encode("")
    assert not np.isnan(encoded).any()
    assert np.all(encoded == 0)


def test_encode_multiple_sentences(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of multiple sentences."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode(["word1 word2", "word1 word3"])
    assert encoded.shape == (2, 2)


def test_encode_as_sequence(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test encoding of sentences as tokens."""
    sentences = ["word1 word2", "word1 word3"]
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded_sequence = model.encode_as_sequence(sentences)
    encoded = model.encode(sentences)

    assert len(encoded_sequence) == 2

    means = [np.mean(sequence, axis=0) for sequence in encoded_sequence]
    assert np.allclose(means, encoded)


def test_encode_multiprocessing(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding with multiprocessing."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    # Generate a list of 15k inputs to test multiprocessing
    sentences = ["word1 word2"] * 15_000
    encoded = model.encode(sentences, use_multiprocessing=True)
    assert encoded.shape == (15000, 2)


def test_encode_as_sequence_multiprocessing(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of sentences as tokens with multiprocessing."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    # Generate a list of 15k inputs to test multiprocessing
    sentences = ["word1 word2"] * 15_000
    encoded = model.encode_as_sequence(sentences, use_multiprocessing=True)
    assert len(encoded) == 15_000


def test_encode_as_tokens_empty(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of an empty list of sentences."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode_as_sequence("")
    assert np.array_equal(encoded, np.zeros(shape=(0, 2), dtype=model.embedding.dtype))

    encoded = model.encode_as_sequence(["", ""])
    out = [np.zeros(shape=(0, 2), dtype=model.embedding.dtype) for _ in range(2)]
    assert [np.array_equal(x, y) for x, y in zip(encoded, out)]


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

    assert (save_path / "model.safetensors").exists()
    assert (save_path / "tokenizer.json").exists()
    assert (save_path / "config.json").exists()
    assert (save_path / "modules.json").exists()


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
    np.testing.assert_array_equal(loaded_model.embedding, mock_vectors)
    assert loaded_model.tokenizer.get_vocab() == mock_tokenizer.get_vocab()
    assert loaded_model.config == mock_config


def test_load_pretrained_quantized(
    tmp_path: Path, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test loading a pretrained model after saving it."""
    # Save the model to a temporary path
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, quantize_to="int8")

    # Assert that the loaded model has the same properties as the original one
    assert loaded_model.embedding.dtype == np.int8
    assert loaded_model.embedding.shape == mock_vectors.shape

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, quantize_to="float16")

    # Assert that the loaded model has the same properties as the original one
    assert loaded_model.embedding.dtype == np.float16
    assert loaded_model.embedding.shape == mock_vectors.shape

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, quantize_to="float32")
    # Assert that the loaded model has the same properties as the original one
    assert loaded_model.embedding.dtype == np.float32
    assert loaded_model.embedding.shape == mock_vectors.shape


def test_load_pretrained_dim(
    tmp_path: Path, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test loading a pretrained model with dimensionality."""
    # Save the model to a temporary path
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    loaded_model = StaticModel.from_pretrained(save_path, dimensionality=2)

    # Assert that the loaded model has the same properties as the original one
    np.testing.assert_array_equal(loaded_model.embedding, mock_vectors[:, :2])
    assert loaded_model.tokenizer.get_vocab() == mock_tokenizer.get_vocab()
    assert loaded_model.config == mock_config

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, dimensionality=None)

    # Assert that the loaded model has the same properties as the original one
    np.testing.assert_array_equal(loaded_model.embedding, mock_vectors)
    assert loaded_model.tokenizer.get_vocab() == mock_tokenizer.get_vocab()
    assert loaded_model.config == mock_config

    # Load the model back from the same path
    with pytest.raises(ValueError):
        StaticModel.from_pretrained(save_path, dimensionality=3000)


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


def test_dim(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Tests the dimensionality of the model."""
    model = StaticModel(mock_vectors, mock_tokenizer, mock_config)
    assert model.dim == 2
    assert model.dim == model.embedding.shape[1]


def test_local_load_from_model(mock_tokenizer: Tokenizer) -> None:
    """Test local load from a model."""
    x = np.ones((mock_tokenizer.get_vocab_size(), 2))
    with TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        safetensors.numpy.save_file({"embeddings": x}, Path(tempdir) / "model.safetensors")
        mock_tokenizer.save(str(Path(tempdir) / "tokenizer.json"))

        model = StaticModel.load_local(tempdir_path)
        assert model.embedding.shape == x.shape
        assert model.tokenizer.to_str() == mock_tokenizer.to_str()
        assert model.config == {"normalize": False}


def test_local_load_from_model_no_folder() -> None:
    """Test local load from a model with no folder."""
    with pytest.raises(ValueError):
        StaticModel.load_local("woahbuddy_relax_this_is_just_a_test")
