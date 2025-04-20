import numpy as np
import pytest

from model2vec.quantization import DType, quantize_embeddings


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
