from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import OPTICS as SKLearnOPTICS

from .base import ClusteringAlgorithm, ClusteringResult


@dataclass(slots=True)
class OpticsConfig:
    min_samples: int
    min_samples_range: tuple[int, int] | None
    metric: str 
    cluster_method: str
    xi: float
    xi_range: tuple[float, float] | None
    n_jobs: Optional[int]
    n_trials: int


class SklearnOpticsAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.OPTICS that implements the project's
    `ClusteringAlgorithm`/`ClusteringResult` interface.
    """

    def __init__(self, config: OpticsConfig, **sk_kwargs) -> None:
        self.config = config
        self._sk = SKLearnOPTICS(
            min_samples=config.min_samples,
            metric=config.metric,
            cluster_method=config.cluster_method,
            xi=config.xi,
            n_jobs=config.n_jobs,
            **sk_kwargs,
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
