from __future__ import annotations

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
        # Normalize to [-128, 127] range for int8
        # We normalize to -127 to 127 to keep symmetry.
        scale = np.max(np.abs(embeddings)) / 127.0
        quantized = np.round(embeddings / scale).astype(np.int8)
        return quantized
    else:
        raise ValueError("Not a valid enum member of DType.")


def quantize_and_reduce_dim(
    embeddings: np.ndarray, quantize_to: str | DType | None, dimensionality: int | None
) -> np.ndarray:
    """
    Quantize embeddings to a datatype and reduce dimensionality.

    :param embeddings: The embeddings to quantize and reduce, as a numpy array.
    :param quantize_to: The data type to quantize to. If None, no quantization is performed.
    :param dimensionality: The number of dimensions to keep. If None, no dimensionality reduction is performed.
    :return: The quantized and reduced embeddings.
    :raises ValueError: If the passed dimensionality is not None and greater than the model dimensionality.
    """
    if quantize_to is not None:
        quantize_to = DType(quantize_to)
        embeddings = quantize_embeddings(embeddings, quantize_to)

    if dimensionality is not None:
        if dimensionality > embeddings.shape[1]:
            raise ValueError(
                f"Dimensionality {dimensionality} is greater than the model dimensionality {embeddings.shape[1]}"
            )
        embeddings = embeddings[:, :dimensionality]

    return embeddings
