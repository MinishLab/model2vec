import numpy as np
import pytest

from model2vec.quantization import DType, quantize_and_reduce_dim, quantize_embeddings
from model2vec.vocabulary_quantization import quantize_vocabulary


@pytest.mark.parametrize(
    "input_dtype,target_dtype,expected_dtype",
    [
        (np.float32, DType.Float16, np.float16),
        (np.float16, DType.Float32, np.float32),
        (np.float32, DType.Float64, np.float64),
        (np.float32, DType.Int8, np.int8),
    ],
)
def test_quantize_embeddings(input_dtype: DType, target_dtype: DType, expected_dtype: DType) -> None:
    """Test quantization to different dtypes."""
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=input_dtype)
    # Use negative values for int8 test case
    if target_dtype == DType.Int8:
        embeddings = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=input_dtype)

    quantized = quantize_embeddings(embeddings, target_dtype)
    assert quantized.dtype == expected_dtype

    if target_dtype == DType.Int8:
        # Check if the values are in the range [-127, 127]
        assert np.all(quantized >= -127) and np.all(quantized <= 127)
    else:
        assert np.allclose(quantized, embeddings.astype(expected_dtype))


def test_quantize_and_reduce_dim() -> None:
    """Test quantization and dimensionality reduction."""
    embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # Test quantization only
    quantized = quantize_and_reduce_dim(embeddings, DType.Float16, None)
    assert quantized.dtype == np.float16

    # Test dimensionality reduction only
    reduced = quantize_and_reduce_dim(embeddings, None, 2)
    assert reduced.shape == (2, 2)

    # Test both quantization and dimensionality reduction
    quantized_reduced = quantize_and_reduce_dim(embeddings, DType.Int8, 2)
    assert quantized_reduced.dtype == np.int8
    assert quantized_reduced.shape == (2, 2)


def test_quantize_vocabulary() -> None:
    """Test quantization of vocabulary."""
    rand = np.random.RandomState(42)
    embeddings = rand.normal(size=(1024, 4)).astype(np.float32)
    norm = np.linalg.norm(embeddings, axis=1)

    # Test without quantization
    quantized_embeddings, _, quantized_weights = quantize_vocabulary(8, None, embeddings)
    assert quantized_embeddings.dtype == np.float32
    assert quantized_embeddings.shape == (8, 4)
    # assert np.array_equal(quantized_token_mapping, token_mapping)
    assert np.array_equal(norm, quantized_weights)

    rand_weights = rand.uniform(size=(1024,))
    # Test without quantization
    quantized_embeddings, _, quantized_weights = quantize_vocabulary(8, rand_weights, embeddings)
    assert quantized_embeddings.dtype == np.float32
    assert quantized_embeddings.shape == (8, 4)
    # Weights should be the same as the in weights
    assert np.array_equal(rand_weights, quantized_weights)

    embeddings = embeddings.astype(np.float16)
    # Test with float16 quantization
    quantized_embeddings, _, quantized_weights = quantize_vocabulary(8, None, embeddings)
    assert quantized_embeddings.dtype == np.float16
