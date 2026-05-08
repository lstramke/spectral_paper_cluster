from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from app_types.optimization_field import OptimizationField

@dataclass(slots=True)
class ClusteringResult:
    """Output contract for any clustering algorithm."""

    labels: torch.Tensor
    n_clusters_found: int
    probabilities: torch.Tensor | None = None
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

@dataclass(slots=True)
class ClusteringConfig(ABC):
    """Marker base class for clustering config dataclasses.

    All clustering config dataclasses in `src.clustering` should subclass
    this to provide a consistent type for factory creation.
    """
    
    @abstractmethod
    def get_optimization_fields(self) -> list[OptimizationField]:
        """Return the optimization fields this config supports.
        
        Subclasses override to declare their own fields.
        """
        pass

    @abstractmethod
    def get_n_trials(self) -> int:
        pass
