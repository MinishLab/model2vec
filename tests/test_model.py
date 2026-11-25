from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from model2vec import StaticModel


def test_initialization(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test successful initialization of StaticModel."""
    original_config = dict(mock_config)
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    assert model.embedding.shape == (5, 2)
    assert len(model.tokens) == 5
    assert model.tokenizer.get_vocab() == mock_tokenizer.get_vocab()
    assert model.config == {**original_config, "normalize": False, "max_length": 512}
    assert mock_config == original_config


def test_initialization_token_vector_mismatch(mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test if error is raised when number of tokens and vectors don't match."""
    mock_vectors = np.array([[0.1, 0.2], [0.2, 0.3]])
    with pytest.raises(ValueError):
        StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)


def test_tokenize(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test tokenization of a sentence."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    tokens = model.tokenize(["word1 word2"])

    assert tokens


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


def test_encode_single_sentence_empty_float16(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of a single empty sentence with float16 embeddings."""
    model = StaticModel(vectors=mock_vectors.astype(np.float16), tokenizer=mock_tokenizer, config=mock_config)
    model.normalize = True
    encoded = model.encode("")
    assert not np.isnan(encoded).any()
    assert np.all(encoded == 0)
    assert encoded.dtype == model.embedding.dtype


def test_encode_multiple_sentences(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of multiple sentences."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded = model.encode(["word1 word2", "word1 word3"])
    assert encoded.shape == (2, 2)


def test_encode_int8_quantized(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test that encoding an int8-quantized model returns float32 output."""
    from model2vec.model import quantize_model

    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    quantized = quantize_model(model, quantize_to="int8")
    assert quantized.embedding.dtype == np.int8

    encoded = quantized.encode(["word1 word2", "word1 word3"])
    assert encoded.dtype == np.float32


def test_encode_as_sequence(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Test encoding of sentences as tokens."""
    sentences = ["word1 word2", "word1 word3"]
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded_sequence = model.encode_as_sequence(sentences)
    encoded = model.encode(sentences)

    assert len(encoded_sequence) == 2

    means = [np.mean(sequence, axis=0) for sequence in encoded_sequence]
    assert np.allclose(means, encoded)


def test_encode_as_sequence_max_length(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of sentences as tokens with max_length truncation."""
    sentences = ["word1 word2 word3"]
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    full = model.encode_as_sequence(sentences)
    truncated = model.encode_as_sequence(sentences, max_length=1)

    assert len(truncated[0]) < len(full[0])


def test_encode_as_tokens_empty(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test encoding of an empty list of sentences."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    encoded: list[np.ndarray] | np.ndarray
    encoded = model.encode_as_sequence("")
    assert np.array_equal(encoded, np.zeros(shape=(0, 2), dtype=model.embedding.dtype))

    encoded_list = model.encode_as_sequence(["", ""])
    out = [np.zeros(shape=(0, 2), dtype=model.embedding.dtype) for _ in range(2)]
    assert [np.array_equal(x, y) for x, y in zip(encoded_list, out)]


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
    for k, v in mock_config.items():
        assert loaded_model.config.get(k) == v
    assert "embedding_dtype" in loaded_model.config


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
    assert loaded_model.embedding_dtype == "int8"

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, quantize_to="float16")

    # Assert that the loaded model has the same properties as the original one
    assert loaded_model.embedding.dtype == np.float16
    assert loaded_model.embedding.shape == mock_vectors.shape
    assert loaded_model.embedding_dtype == "float16"

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, quantize_to="float32")
    # Assert that the loaded model has the same properties as the original one
    assert loaded_model.embedding.dtype == np.float32
    assert loaded_model.embedding.shape == mock_vectors.shape
    assert loaded_model.embedding_dtype == "float32"

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, quantize_to="float64")
    # Assert that the loaded model has the same properties as the original one
    assert loaded_model.embedding.dtype == np.float64
    # Should not copy if same as original.
    assert loaded_model.embedding is loaded_model.embedding


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
    for k, v in mock_config.items():
        assert loaded_model.config.get(k) == v
    assert "embedding_dtype" in loaded_model.config

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, dimensionality=None)

    # Assert that the loaded model has the same properties as the original one
    np.testing.assert_array_equal(loaded_model.embedding, mock_vectors)
    assert loaded_model.tokenizer.get_vocab() == mock_tokenizer.get_vocab()
    for k, v in mock_config.items():
        assert loaded_model.config.get(k) == v
    assert "embedding_dtype" in loaded_model.config

    # Load the model back from the same path
    with pytest.raises(ValueError):
        StaticModel.from_pretrained(save_path, dimensionality=3000)


def test_load_pretrained_vocabulary_quantized(
    tmp_path: Path, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]
) -> None:
    """Test loading a pretrained model with vocabulary quantization."""
    # Save the model to a temporary path
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, config=mock_config)
    save_path = tmp_path / "saved_model"
    model.save_pretrained(save_path)

    # Load the model back from the same path
    loaded_model = StaticModel.from_pretrained(save_path, vocabulary_quantization=3)

    # Assert that the loaded model has the same properties as the original one
    assert loaded_model.embedding.dtype == np.float64
    assert loaded_model.embedding.shape == (3, 2)  # Assuming 3 clusters after quantization
    assert loaded_model.weights is not None
    assert loaded_model.weights.shape == (5,)
    assert loaded_model.token_mapping is not None
    assert loaded_model.vocabulary_quantization == 3
    assert len(loaded_model.token_mapping) == mock_tokenizer.get_vocab_size()
    assert len(loaded_model.token_mapping) == len(loaded_model.weights)
    assert loaded_model.encode("word1 word2").shape == (2,)
    assert loaded_model.encode(["word1 word2"] * 4).shape == (4, 2)


def test_initialize_normalize(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Tests whether the normalization initialization is correct."""
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
    assert model.config == {"normalize": False, "max_length": 512}
    model.normalize = True
    assert model.config == {"normalize": True, "max_length": 512}


def test_set_max_length(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Tests whether the max_length is set correctly."""
    model = StaticModel(mock_vectors, mock_tokenizer, {}, max_length=128)
    assert model.config == {"normalize": False, "max_length": 128}
    assert model.tokenizer.truncation["max_length"] == 128
    model.max_length = 256
    assert model.config == {"normalize": False, "max_length": 256}
    assert model.tokenizer.truncation["max_length"] == 256
    model.max_length = None
    assert model.config == {"normalize": False, "max_length": None}
    assert model.tokenizer.truncation is None


def test_models_do_not_share_tokenizer(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Tests that two models built from the same tokenizer object have independent truncation state."""
    model_a = StaticModel(mock_vectors, mock_tokenizer, {}, max_length=128)
    model_b = StaticModel(mock_vectors, mock_tokenizer, {}, max_length=256)

    model_b.max_length = 8

    assert model_a.tokenizer.truncation["max_length"] == 128
    assert model_b.tokenizer.truncation["max_length"] == 8


def test_dim(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, mock_config: dict[str, str]) -> None:
    """Tests the dimensionality of the model."""
    model = StaticModel(mock_vectors, mock_tokenizer, mock_config)
    assert model.dim == 2
    assert model.dim == model.embedding.shape[1]


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("max_length", [None, 1])
def test_encode_matches_sequence_mean(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, normalize: bool, max_length: int | None
) -> None:
    """Test that encode matches the mean of encode_as_sequence, with weights and a token mapping."""
    weights = np.linspace(0.5, 1.5, len(mock_vectors)).astype(np.float32)
    token_mapping = np.array([0, 1, 2, 1, 0])
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, weights=weights, token_mapping=token_mapping)
    sentences = ["word1 word2", "word3 word1 word2", "", "unknown"]
    encoded = model.encode(sentences, normalize=normalize, max_length=max_length)
    expected = np.stack(
        [
            seq.mean(axis=0) if len(seq) else np.zeros(model.dim)
            for seq in model.encode_as_sequence(sentences, max_length=max_length)
        ]
    )
    if normalize:
        expected = expected / (np.linalg.norm(expected, axis=1, keepdims=True) + 1e-32)
    assert encoded.dtype == model.embedding.dtype
    np.testing.assert_allclose(encoded, expected, rtol=1e-5, atol=1e-7)


def test_encode_rust_backend_rebuilds(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test that the Rust backend is cached and rebuilt when the embedding is replaced."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    before = model.encode(["word1"])
    backend = model._get_rust_backend()
    assert model._get_rust_backend() is backend
    model.embedding = model.embedding * 2
    assert model._get_rust_backend() is not backend
    np.testing.assert_allclose(model.encode(["word1"]), before * 2)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int8])
def test_arrays_are_read_only(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, dtype: type) -> None:
    """Test that the model arrays cannot be changed in place, before or after encoding, and only by assignment."""
    import pickle

    model = StaticModel(
        vectors=(mock_vectors * 10).astype(dtype),
        tokenizer=mock_tokenizer,
        weights=np.ones(len(mock_vectors), dtype=np.float32),
        token_mapping=np.arange(len(mock_vectors)),
    )
    for clone in (model, pickle.loads(pickle.dumps(model))):
        for array in (clone.embedding, clone.weights, clone.token_mapping):
            with pytest.raises(ValueError, match="read-only"):
                array *= 2
    before = model.encode(["word1"])
    with pytest.raises(ValueError, match="read-only"):
        model.embedding *= 2
    model.embedding = model.embedding * 2
    with pytest.raises(ValueError, match="read-only"):
        model.embedding[0] = 0
    np.testing.assert_allclose(model.encode(["word1"]), before * 2)
    np.testing.assert_allclose(model.encode(["word1"])[0], model.encode_as_sequence(["word1"])[0].mean(axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.float16, np.float64])
def test_encode_shares_arrays_with_rust_backend(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, dtype: type
) -> None:
    """Test that encoding swaps the arrays for views of the Rust backend's memory when their dtypes match."""
    original = (mock_vectors * 10).astype(dtype)
    weights = np.ones(len(mock_vectors), dtype=np.float32)
    model = StaticModel(vectors=original, tokenizer=mock_tokenizer, weights=weights)
    expected = model.encode(["word1 word2"])
    backend = model._get_rust_backend()

    shared = dtype in (np.float32, np.int8)
    assert np.shares_memory(model.embedding, backend.embeddings) == shared
    assert np.shares_memory(model.embedding, original) != shared
    assert np.shares_memory(model.weights, backend.weights)
    assert model.embedding.dtype == dtype
    np.testing.assert_array_equal(model.embedding, original)
    np.testing.assert_array_equal(model.weights, weights)
    assert model._get_rust_backend() is backend
    np.testing.assert_array_equal(model.encode(["word1 word2"]), expected)
    with pytest.raises(ValueError, match="read-only"):
        model.embedding[0] = 0


def test_input_arrays_stay_writable(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test that the model does not make the arrays it is given read-only."""
    vectors = mock_vectors.copy()
    weights = np.ones(len(vectors), dtype=np.float32)
    token_mapping = np.arange(len(vectors))
    model = StaticModel(vectors=vectors, tokenizer=mock_tokenizer, weights=weights, token_mapping=token_mapping)
    model.encode(["word1"])
    for array in (vectors, weights, token_mapping):
        assert array.flags.writeable
        array[0] = 0


def test_encode_rust_backend_rebuilds_on_tokenizer_assignment(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Test that assigning the tokenizer rebuilds the Rust backend, so in-place changes are used."""
    from tokenizers.normalizers import Lowercase

    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    before = model.encode(["WORD1"])
    backend = model._get_rust_backend()
    model.tokenizer.normalizer = Lowercase()
    model.tokenizer = model.tokenizer
    assert model._get_rust_backend() is not backend
    after = model.encode(["WORD1"])
    assert not np.allclose(before, after)
    np.testing.assert_allclose(after, model.encode(["word1"]))


@pytest.mark.parametrize("method", ["encode", "encode_as_sequence"])
def test_encode_removed_arguments_warn(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, method: str) -> None:
    """Test that the removed encoding arguments raise a DeprecationWarning, and other arguments do not."""
    import warnings

    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    with pytest.warns(DeprecationWarning, match="`batch_size`, `show_progress_bar`"):
        getattr(model, method)(["word1"], show_progress_bar=True, batch_size=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        getattr(model, method)(["word1"], something_else=True)


def test_pickle_and_copy_with_rust_backend(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test that a model with a built Rust backend can be pickled and copied."""
    import copy
    import pickle

    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    expected = model.encode(["word1 word2"])
    for clone in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model), copy.copy(model)):
        np.testing.assert_allclose(clone.encode(["word1 word2"]), expected)
