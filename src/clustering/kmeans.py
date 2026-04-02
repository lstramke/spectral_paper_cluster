from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from .base import ClusteringAlgorithm, ClusteringResult


class KMeans(ClusteringAlgorithm):
    def __init__(
        self,
        n_clusters: int,
        max_iter: int,
        tol: float,
        seed: int,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if tol < 0:
            raise ValueError("tol must be >= 0")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.centroids_: Tensor | None = None
        self.labels_: Tensor | None = None
        self.inertia_: float | None = None

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        # KMeans expects a feature matrix with shape (n_samples, n_features).
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        if x.size(0) < self.n_clusters:
            raise ValueError("number of samples must be >= n_clusters")

        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.seed)

        # Initialize centroids by sampling k points from the dataset.
        perm = torch.randperm(x.size(0), generator=generator, device=x.device)
        centroids = x[perm[: self.n_clusters]].clone()

        for _ in range(self.max_iter):
            # Assign each sample to its nearest centroid.
            distances = torch.cdist(x, centroids)
            labels = torch.argmin(distances, dim=1)

            new_centroids: list[Tensor] = []
            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                if torch.any(mask):
                    # Update centroid as mean of all assigned samples.
                    new_centroids.append(x[mask].mean(dim=0))
                else:
                    # Keep centroid if cluster is empty to avoid invalid updates.
                    new_centroids.append(centroids[cluster_id])

            next_centroids = torch.stack(new_centroids, dim=0)
            diff = next_centroids - centroids
            shift = cast(float, torch.sum(diff * diff).sqrt().item())
            centroids = next_centroids

            # Stop if centroid movement is smaller than tolerance.
            if shift <= self.tol:
                break

        # Recompute labels and objective using final centroids.
        final_distances = torch.cdist(x, centroids)
        final_labels = torch.argmin(final_distances, dim=1)
        min_distances = torch.gather(final_distances, 1, final_labels.unsqueeze(1)).squeeze(1)
        inertia = float(torch.sum(min_distances ** 2).item())

        self.centroids_ = centroids
        self.labels_ = final_labels
        self.inertia_ = inertia
        return self

    def predict(self, x: Tensor) -> Tensor:
        if self.centroids_ is None:
            raise RuntimeError("KMeans is not fitted. Call fit() first.")
        distances = torch.cdist(x, self.centroids_)
        return torch.argmin(distances, dim=1)

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        self.fit(x)
        if self.labels_ is None:
            raise RuntimeError("KMeans fit failed to produce labels")

        cluster_sizes: dict[int, int] = {}
        for value in self.labels_.detach().cpu().flatten():
            cluster_id = int(value.item())
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        return ClusteringResult(
            labels=self.labels_,
            n_clusters_found=len(cluster_sizes),
            centroids=self.centroids_,
            objective=self.inertia_,
            cluster_sizes=cluster_sizes,
            metadata={
                "algorithm": "kmeans",
                "n_clusters": self.n_clusters,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "seed": self.seed,
            },
        )