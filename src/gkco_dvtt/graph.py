from typing import Tuple

import numpy as np
import torch
from scipy.fft import fft
from sklearn.metrics.pairwise import euclidean_distances


def _ensure_2d(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)

    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    if samples.ndim > 2:
        samples = samples.reshape(samples.shape[0], -1)

    return samples


def fft_half_spectrum(samples: np.ndarray) -> np.ndarray:
    """
    Convert raw signals to half-spectrum FFT features.
    """
    samples = _ensure_2d(samples)

    fft_features = np.abs(fft(samples, axis=1))
    half_length = samples.shape[1] // 2

    return fft_features[:, :half_length].astype(np.float32)


def _neighbors_to_edge_index(neighbors: np.ndarray) -> torch.Tensor:
    num_nodes, k = neighbors.shape

    source = np.repeat(np.arange(num_nodes), k)
    target = neighbors.reshape(-1)

    edge_index = np.stack([source, target], axis=0)

    return torch.tensor(edge_index, dtype=torch.long)


def build_gkco_graphs(
    node_features: np.ndarray,
    k_original: int,
    k_positive: int = 1,
    k_negative: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct original, positive, and negative KNN graphs."""
    distances = euclidean_distances(node_features)

    # Nearest-neighbor ranking
    nearest_index = np.argsort(distances, axis=1)

    original_neighbors = nearest_index[:, 1:1 + k_original]
    positive_neighbors = nearest_index[:, 1:1 + k_positive]

    farthest_index = np.argsort(-distances, axis=1)
    negative_neighbors = farthest_index[:, 1:1 + k_negative]

    edge_original = _neighbors_to_edge_index(original_neighbors)
    edge_positive = _neighbors_to_edge_index(positive_neighbors)
    edge_negative = _neighbors_to_edge_index(negative_neighbors)

    return edge_original, edge_positive, edge_negative


def build_rank_weighted_graph(
    node_features: np.ndarray,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a KNN graph with cosine-similarity edge weights."""

    node_features = np.asarray(
        node_features,
        dtype=np.float32,
    )

    num_nodes = node_features.shape[0]

    # KNN topology reconstruction

    distances = euclidean_distances(
        node_features
    )

    nearest_index = np.argsort(
        distances,
        axis=1,
    )

    neighbors = nearest_index[
        :,
        1:1 + k,
    ]

    adjacency = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float32,
    )

    weights = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float32,
    )

    # Cosine-similarity edge weights

    eps = 1e-12

    norms = np.linalg.norm(
        node_features,
        axis=1,
    )

    for i in range(num_nodes):

        neighbor_ids = neighbors[i]

        similarities = []

        for neighbor in neighbor_ids:

            numerator = np.dot(
                node_features[i],
                node_features[neighbor],
            )

            denominator = (
                norms[i]
                * norms[neighbor]
                + eps
            )

            similarity = (
                numerator
                / denominator
            )

            similarities.append(
                similarity
            )

        similarities = np.asarray(
            similarities,
            dtype=np.float32,
        )

        # Numerically stable exp-normalization
        similarities = (
            similarities
            - np.max(similarities)
        )

        exp_similarity = np.exp(
            similarities
        )

        normalized_weights = (
            exp_similarity
            / (
                np.sum(exp_similarity)
                + eps
            )
        )

        for j, neighbor in enumerate(
            neighbor_ids
        ):

            weight = normalized_weights[j]

            adjacency[
                i,
                neighbor,
            ] = weight

            weights[
                i,
                neighbor,
            ] = weight

    return (
        torch.tensor(
            adjacency,
            dtype=torch.float32,
        ),
        torch.tensor(
            weights,
            dtype=torch.float32,
        ),
    )