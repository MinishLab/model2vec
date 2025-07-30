from __future__ import annotations

from enum import Enum
from typing import cast

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


def quantize_vocabulary(
    n_clusters: int, weights: np.ndarray | None, embeddings: np.ndarray
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Quantize the vocabulary of embeddings using KMeans clustering."""
    # If the model does not have weights, we assume the norm to be informative.
    if weights is None:
        weights = cast(np.ndarray, np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-32)
        # Divide by the norm to normalize the embeddings, so we don't bias the clustering.
        embeddings = embeddings / weights

    # Quantize the vocabulary
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    # Create a mapping from the original token index to the cluster index
    # Make sure to convert to list, otherwise we get np.int32 which is not jsonable.
    token_mapping = cast(list[int], kmeans.predict(embeddings).tolist())
    # The cluster centers are the new embeddings.
    embeddings = kmeans.cluster_centers_

    return embeddings, token_mapping, weights
