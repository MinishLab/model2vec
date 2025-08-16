import logging
from typing import cast

import numpy as np

# Lazy import
try:
    from sklearn.cluster import KMeans
except ImportError:
    raise ImportError(
        "scikit-learn is required for quantizing the vocabulary. "
        "Please install model2vec with the quantization extra."
    )


logger = logging.getLogger(__name__)


def quantize_vocabulary(
    n_clusters: int, weights: np.ndarray | None, embeddings: np.ndarray
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Quantize the vocabulary of embeddings using KMeans clustering."""
    logger.info(f"Quantizing vocabulary to {n_clusters} clusters.")
    # If the model does not have weights, we assume the norm to be informative.
    if weights is None:
        weights = cast(np.ndarray, np.linalg.norm(embeddings, axis=1) + 1e-32)
        # Divide by the norm to normalize the embeddings, so we don't bias the clustering.
        embeddings = embeddings / weights[:, None]

    # Ensure the embeddings are in float32 for KMeans
    # Store the original dtype to restore it later
    orig_dtype = embeddings.dtype

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="random")
    cast_embeddings = embeddings.astype(np.float32)
    # Fit KMeans to the embeddings
    kmeans.fit(cast_embeddings)
    # Create a mapping from the original token index to the cluster index
    # Make sure to convert to list, otherwise we get np.int32 which is not jsonable.
    token_mapping = cast(list[int], kmeans.predict(cast_embeddings).tolist())
    # The cluster centers are the new embeddings.
    # Convert them back to the original dtype
    embeddings = kmeans.cluster_centers_.astype(orig_dtype)

    return embeddings, token_mapping, weights
