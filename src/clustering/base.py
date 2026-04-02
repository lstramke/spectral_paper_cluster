from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

@dataclass(slots=True)
class ClusteringResult:
    """Output contract for any clustering algorithm."""

    labels: torch.Tensor
    n_clusters_found: int
    centroids: torch.Tensor | None = None
    objective: float | None = None
    cluster_sizes: dict[int, int] = field(default_factory=dict[int, int])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class ClusteringAlgorithm(ABC):
    """Common interface for all clustering algorithms in this project."""

    @abstractmethod
    def fit(self, x: torch.Tensor) -> ClusteringAlgorithm:
        """Fit the model to feature matrix x."""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict cluster ids for feature matrix x.

        Some clustering algorithms do not support out-of-sample prediction.
        """
        raise NotImplementedError("predict() is not supported for this algorithm")

    def fit_predict(self, x: torch.Tensor) -> ClusteringResult:
        """Fit and return a standardized clustering result."""
        self.fit(x)
        labels = self.predict(x)
        cluster_sizes: dict[int, int] = {}
        for value in labels.detach().cpu().flatten():
            cluster_id = int(value.item())
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        return ClusteringResult(
            labels=labels,
            n_clusters_found=len(cluster_sizes),
            cluster_sizes=cluster_sizes,
        )

