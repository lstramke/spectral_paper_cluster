from __future__ import annotations

import torch


def make_synthetic_blobs(
    n_samples: int = 600,
    n_features: int = 2,
    n_clusters: int = 3,
    std: float = 0.8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic Gaussian blobs for clustering demos."""
    if n_samples < n_clusters:
        raise ValueError("n_samples must be >= n_clusters")

    generator = torch.Generator()
    generator.manual_seed(seed)

    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters

    centers = torch.randn(n_clusters, n_features, generator=generator) * 5.0

    xs = []
    ys = []
    for cluster_id in range(n_clusters):
        count = samples_per_cluster + (1 if cluster_id < remainder else 0)
        points = centers[cluster_id] + std * torch.randn(count, n_features, generator=generator)
        xs.append(points)
        ys.append(torch.full((count,), cluster_id, dtype=torch.long))

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y
