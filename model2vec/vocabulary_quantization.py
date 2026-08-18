from __future__ import annotations

import logging
from typing import cast

import numpy as np

# Lazy import
try:
    from usearch.index import Index
except ImportError:
    raise ImportError(
        "usearch is required for quantizing the vocabulary. Please install model2vec with the quantization extra."
    )


logger = logging.getLogger(__name__)


def quantize_vocabulary(
    embeddings: np.ndarray,
    weights: np.ndarray | None,
    sim_threshold: float = 0.55,
    drop_fraction: float = 0.2,
    n_neighbors: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize the vocabulary by merging near-duplicate and low-weight tokens.

    Tokens are merged in two stages. First, tokens whose cosine similarity exceeds
    `sim_threshold` are merged together (complete-linkage-safe: a group only merges if every
    pair inside it clears the threshold, which avoids chaining unrelated tokens together
    through a string of intermediate near-duplicates). Second, the `drop_fraction` lowest-weight
    resulting clusters are folded into their nearest surviving cluster, regardless of similarity.

    :param embeddings: The embeddings to quantize, as a numpy array.
    :param weights: Per-token importance weights. If None, the embedding norm is used instead,
        and the embeddings are treated as norm-carries-weight (direction is clustered, magnitude is weight).
    :param sim_threshold: Tokens with cosine similarity above this value are eligible to merge.
    :param drop_fraction: Fraction of the resulting clusters (by total weight) to fold into their
        nearest surviving cluster after merging. Set to 0 to skip this stage.
    :param n_neighbors: Number of neighbors considered per token when building the merge candidate graph.
    :return: The merged embeddings, the token-to-cluster mapping, and the per-token weights.
    """
    logger.info(f"Quantizing vocabulary with sim_threshold={sim_threshold}, drop_fraction={drop_fraction}.")

    orig_dtype = embeddings.dtype
    norms = cast(np.ndarray, np.linalg.norm(embeddings, axis=1) + 1e-32)
    if weights is None:
        weights = norms
    normed = (embeddings / norms[:, None]).astype(np.float32)

    labels = _merge_similar_tokens(normed, sim_threshold=sim_threshold, n_neighbors=n_neighbors)
    if drop_fraction > 0:
        labels = _drop_low_weight_clusters(normed, weights, labels, drop_fraction=drop_fraction)

    new_embeddings = _spherical_centroids(normed, labels, weights).astype(orig_dtype)
    return new_embeddings, labels, weights / weights.max()


def _merge_similar_tokens(normed: np.ndarray, sim_threshold: float, n_neighbors: int) -> np.ndarray:
    """Cluster unit-normalized embeddings via a scalable complete-linkage-safe merge.

    Zero vectors have no defined direction, so cosine similarity cannot merge them; they're kept as singletons.
    """
    n = len(normed)
    nonzero_idx = np.flatnonzero(np.any(normed != 0, axis=1))
    labels = np.arange(n)

    if len(nonzero_idx) <= 1:
        return labels

    sub_labels = _bounded_merge(normed[nonzero_idx], sim_threshold=sim_threshold, n_neighbors=n_neighbors)
    # Offset so cluster ids can't collide with a zero vector's singleton (index-based) label.
    labels[nonzero_idx] = sub_labels + n

    _, compact = np.unique(labels, return_inverse=True)
    return compact


def _bounded_merge(vectors: np.ndarray, sim_threshold: float, n_neighbors: int, max_rounds: int = 10) -> np.ndarray:
    """Merge points whose cosine similarity clears `sim_threshold`."""
    n = len(vectors)
    unique_vectors, inverse = np.unique(np.round(vectors, decimals=5), axis=0, return_inverse=True)
    if len(unique_vectors) == 1:
        return np.zeros(n, dtype=np.int64)

    unique_labels = _bounded_merge_unique(
        unique_vectors, sim_threshold=sim_threshold, n_neighbors=n_neighbors, max_rounds=max_rounds
    )
    return cast(np.ndarray, unique_labels[inverse])


def _find(parent: np.ndarray, x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _nearest_neighbor_candidates(root_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find each root's k nearest neighbors among the other roots.

    Returns (row, col, angle) triples, sorted by increasing angle so the closest pairs come first.
    """
    n_roots, dim = root_vectors.shape
    index = Index(ndim=dim, metric="cos")
    index.add(np.arange(n_roots), root_vectors)
    matches = index.search(root_vectors, count=k + 1)

    rows = np.repeat(np.arange(n_roots), matches.keys.shape[1])
    cols = matches.keys.ravel()
    dists = matches.distances.ravel()  # usearch's "cos" metric returns 1 - cosine similarity
    keep = rows != cols
    rows, cols, dists = rows[keep], cols[keep], dists[keep]

    angles = np.arccos(np.clip(1.0 - dists, -1.0, 1.0))
    order = np.argsort(angles)
    return rows[order], cols[order], angles[order]


def _try_merge(
    i: int,
    j: int,
    angle: float,
    theta_max: float,
    parent: np.ndarray,
    radius: np.ndarray,
    mass: np.ndarray,
    centroid: np.ndarray,
) -> bool:
    """Merge cluster j into cluster i in place, if doing so keeps the cluster within theta_max."""
    if radius[i] + radius[j] + angle > theta_max:
        return False

    new_mass = mass[i] + mass[j]
    new_centroid = (mass[i] * centroid[i] + mass[j] * centroid[j]) / new_mass
    norm = np.linalg.norm(new_centroid)
    if norm > 0:
        new_centroid = new_centroid / norm

    angle_i = np.arccos(np.clip(float(new_centroid @ centroid[i]), -1.0, 1.0))
    angle_j = np.arccos(np.clip(float(new_centroid @ centroid[j]), -1.0, 1.0))
    radius[i] = max(radius[i] + angle_i, radius[j] + angle_j)
    centroid[i] = new_centroid
    mass[i] = new_mass
    parent[j] = i
    return True


def _bounded_merge_unique(vectors: np.ndarray, sim_threshold: float, n_neighbors: int, max_rounds: int) -> np.ndarray:
    """Merge points into clusters, each represented by a running centroid."""
    n, dim = vectors.shape
    theta_max = np.arccos(np.clip(sim_threshold, -1.0, 1.0))

    parent = np.arange(n)
    radius = np.zeros(n, dtype=np.float64)
    mass = np.ones(n, dtype=np.float64)
    centroid = vectors.astype(np.float64)

    roots = np.arange(n)
    for _ in range(max_rounds):
        if len(roots) <= 1:
            break
        k = min(n_neighbors, len(roots) - 1)
        root_vectors = centroid[roots].astype(np.float32)
        rows, cols, cand_angle = _nearest_neighbor_candidates(root_vectors, k)
        if len(rows) == 0:
            break
        cand_i, cand_j = roots[rows], roots[cols]

        n_merged = 0
        for idx in range(len(cand_i)):
            # We can only merge roots.
            i, j = int(cand_i[idx]), int(cand_j[idx])
            if parent[i] != i or parent[j] != j:
                continue
            # Do a merge in place, and record whether we did one.
            n_merged += _try_merge(i, j, cand_angle[idx], theta_max, parent, radius, mass, centroid)

        # Quit if we didn't merge anything.
        if n_merged == 0:
            break
        roots = np.unique(np.fromiter((_find(parent, int(r)) for r in roots), dtype=np.int64, count=len(roots)))

    labels = np.fromiter((_find(parent, i) for i in range(n)), dtype=np.int64, count=n)
    _, compact = np.unique(labels, return_inverse=True)
    return cast(np.ndarray, compact)


def _drop_low_weight_clusters(
    normed: np.ndarray, weights: np.ndarray, labels: np.ndarray, drop_fraction: float
) -> np.ndarray:
    """Fold the lowest-total-weight resulting clusters into their nearest surviving cluster."""
    n_clusters = int(labels.max()) + 1
    n_drop = int(n_clusters * drop_fraction)
    if n_drop == 0:
        return labels

    cluster_weight = np.bincount(labels, weights=weights, minlength=n_clusters)
    centroids = _spherical_centroids(normed, labels)

    order = np.argsort(cluster_weight)
    drop_ids = order[:n_drop]
    keep_ids = order[n_drop:]

    index = Index(ndim=centroids.shape[1], metric="cos")
    index.add(keep_ids, centroids[keep_ids])
    matches = index.search(centroids[drop_ids], count=1)
    nearest_keep = matches.keys.reshape(-1)

    remap = np.arange(n_clusters)
    remap[drop_ids] = nearest_keep

    _, compact = np.unique(remap[labels], return_inverse=True)
    return cast(np.ndarray, compact)


def _spherical_centroids(normed: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Compute the (optionally weight-weighted) mean direction of each cluster."""
    n_clusters = int(labels.max()) + 1
    dim = normed.shape[1]
    sums = np.zeros((n_clusters, dim), dtype=np.float64)
    if weights is None:
        np.add.at(sums, labels, normed)
        totals = np.maximum(np.bincount(labels, minlength=n_clusters), 1)
    else:
        np.add.at(sums, labels, normed * weights[:, None])
        totals = np.maximum(np.bincount(labels, weights=weights, minlength=n_clusters), 1e-32)
    return sums / totals[:, None]
