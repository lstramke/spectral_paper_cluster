from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import OPTICS as SKLearnOPTICS

from app_types.optimization_field import OptimizationField

from .base import ClusteringAlgorithm, ClusteringResult, ClusteringConfig


@dataclass(slots=True)
class OpticsConfig(ClusteringConfig):
    min_samples: int
    min_samples_range: tuple[int, int] | None
    metric: str 
    cluster_method: str
    xi: float
    xi_range: tuple[float, float] | None
    n_jobs: Optional[int]
    n_trials: int

    def get_n_trials(self) -> int:
        return self.n_trials

    def get_optimization_fields(self) -> list[OptimizationField]:
        fields: list[OptimizationField] = []

        min_samples_min, min_samples_max = self.min_samples_range or (self.min_samples, self.min_samples)
        fields.append(
            OptimizationField[int](
                name="min_samples",
                min_value=min_samples_min,
                max_value=min_samples_max,
                value_type=int
            )
        )

        xi_min, xi_max = self.xi_range or (self.xi, self.xi)
        fields.append(
            OptimizationField(
                name="xi",
                min_value=xi_min,
                max_value=xi_max,
                value_type=float
            )
        )

        return fields


class SklearnOpticsAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.OPTICS that implements the project's
    `ClusteringAlgorithm`/`ClusteringResult` interface.
    """

    def __init__(self, config: OpticsConfig) -> None:
        self.config = config
        self._sk = SKLearnOPTICS(
            min_samples=config.min_samples,
            metric=config.metric,
            cluster_method=config.cluster_method,
            xi=config.xi,
            n_jobs=config.n_jobs,
        )

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        self._sk.fit(arr)
        return self

    def predict(self, x: Tensor) -> Tensor:
        # OPTICS does not support out-of-sample prediction in sklearn; use
        # fit_predict for training data. Raise to match project expectations.
        raise NotImplementedError("predict() is not supported for OPTICS")

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        labels = self._sk.fit_predict(arr)

        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        # compute cluster sizes and number of clusters (exclude noise label -1)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts) if c > 0}
        n_clusters = len([u for u in unique if int(u) != -1])

        return ClusteringResult(
            labels=labels_t,
            n_clusters_found=int(n_clusters),
            centroids=None,
            objective=None,
            cluster_sizes=cluster_sizes,
            metadata={
                "algorithm": "sklearn_optics",
                "min_samples": self.config.min_samples,
                "metric": self.config.metric,
                "cluster_method": self.config.cluster_method,
                "xi": self.config.xi,
                "n_jobs": self.config.n_jobs,
            },
        )
