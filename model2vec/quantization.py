from __future__ import annotations

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class DType(str, Enum):
    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"
    Int8 = "int8"


dtype_map = {
    DType.Float16: np.float16,
    DType.Float32: np.float32,
    DType.Float64: np.float64,
    DType.Int8: np.int8,
}


def quantize_embeddings(embeddings: np.ndarray, quantize_to: DType) -> np.ndarray:
    """
    Quantize embeddings to a specified data type to reduce memory usage.

    :param embeddings: The embeddings to quantize, as a numpy array.
    :param quantize_to: The data type to quantize to.
    :return: The quantized embeddings.
    :raises ValueError: If the quantization type is not valid.
    """
    mapped_dtype = dtype_map[quantize_to]
    if embeddings.dtype == mapped_dtype:
        # Don't do anything if they match
        return embeddings

    # Handle float types
    if quantize_to in {DType.Float16, DType.Float32, DType.Float64}:
        return embeddings.astype(mapped_dtype)
    elif quantize_to == DType.Int8:
        # Normalize to [-128, 127] range for int8
        # We normalize to -127 to 127 to keep symmetry.
        scale = np.max(np.abs(embeddings)) / 127.0
        # Turn into float16 to minimize memory usage during computation
        # we copy once.
        buf = embeddings.astype(np.float16, copy=True)
        # Divide by the scale
        np.divide(buf, scale, out=buf)
        # Round to int, copy to the buffer
        np.rint(buf, out=buf)
        # Clip to int8 range and convert to int8
        np.clip(buf, -127, 127, out=buf)
        quantized = buf.astype(np.int8)
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
