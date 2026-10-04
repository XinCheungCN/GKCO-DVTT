from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import Data


@dataclass
class GraphTriplet:
    original: Data
    positive: Data
    negative: Data


def half_spectrum(samples: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.fft(samples, axis=1))
    return spectrum[:, : samples.shape[1] // 2].astype(np.float32)


def _edge_index_from_neighbors(neighbors: np.ndarray) -> torch.Tensor:
    sources = np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])
    targets = neighbors.reshape(-1)
    return torch.tensor(np.stack([sources, targets]), dtype=torch.long)


def build_graph_triplet(features: np.ndarray, k_original: int, k_positive: int = 1, k_negative: int = 1) -> GraphTriplet:
    distances = euclidean_distances(features)
    nearest = np.argsort(distances, axis=1)
    farthest = np.argsort(-distances, axis=1)

    original_neighbors = nearest[:, 1:1 + k_original]
    positive_neighbors = nearest[:, 1:1 + k_positive]
    negative_neighbors = farthest[:, :k_negative]

    x = torch.tensor(features, dtype=torch.float32)
    return GraphTriplet(
        original=Data(x=x, edge_index=_edge_index_from_neighbors(original_neighbors)),
        positive=Data(x=x.clone(), edge_index=_edge_index_from_neighbors(positive_neighbors)),
        negative=Data(x=x.clone(), edge_index=_edge_index_from_neighbors(negative_neighbors)),
    )


def build_weighted_optimized_graph(features: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild KNN topology and compute row-normalized cosine-similarity edge weights."""
    x_np = features.detach().cpu().numpy()
    distances = euclidean_distances(x_np)
    nearest = np.argsort(distances, axis=1)[:, 1:1 + k]
    n = features.size(0)

    adjacency = torch.zeros((n, n), dtype=features.dtype, device=features.device)
    weighted_adjacency = torch.zeros_like(adjacency)

    normalized = torch.nn.functional.normalize(features, p=2, dim=1)
    for i in range(n):
        nbr = torch.tensor(nearest[i], dtype=torch.long, device=features.device)
        sims = torch.sum(normalized[i].unsqueeze(0) * normalized[nbr], dim=1)
        weights = torch.softmax(sims, dim=0)
        adjacency[i, nbr] = 1.0
        weighted_adjacency[i, nbr] = weights

    return adjacency, weighted_adjacency
