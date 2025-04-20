from enum import Enum

import numpy as np


class DType(str, Enum):
    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"
    Int8 = "int8"


def quantize_embeddings(embeddings: np.ndarray, quantize_to: DType) -> np.ndarray:
    """
    Quantize embeddings to a specified data type to reduce memory usage.

    :param embeddings: The embeddings to quantize, as a numpy array.
    :param quantize_to: The data type to quantize to.
    :return: The quantized embeddings.
    :raises ValueError: If the quantization type is not valid.
    """
    if quantize_to == DType.Float16:
        return embeddings.astype(np.float16)
    elif quantize_to == DType.Float32:
        return embeddings.astype(np.float32)
    elif quantize_to == DType.Float64:
        return embeddings.astype(np.float64)
    elif quantize_to == DType.Int8:
        # Normalize to [-127, 127] range for int8
        scale = np.max(np.abs(embeddings)) / 127.0
        quantized = np.round(embeddings / scale).astype(np.int8)
        return quantized
    else:
        raise ValueError("Not a valid enum member of DType.")
