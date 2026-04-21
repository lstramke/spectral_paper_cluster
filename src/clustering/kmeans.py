from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import KMeans as SKLearnKMeans

from .base import ClusteringAlgorithm, ClusteringResult

@dataclass(slots=True)
class KMeansConfig:
    n_clusters: int
    max_iter: int
    tol: float
    seed: int
    seed_range: tuple[int, int] | None
    n_trials: int

class SklearnKMeansAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.KMeans that implements the project's
    `ClusteringAlgorithm`/`ClusteringResult` interface.
    """

    def __init__(self, config: KMeansConfig, **sk_kwargs) -> None:
        self.config = config
        self._sk = SKLearnKMeans(
            n_clusters=config.n_clusters,
            max_iter=config.max_iter,
            tol=config.tol,
            random_state=config.seed,
            **sk_kwargs,
        )
        self.n_clusters = config.n_clusters

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        self._sk.fit(arr)
        return self

    def predict(self, x: Tensor) -> Tensor:
        arr = x.detach().cpu().numpy()
        labels = self._sk.predict(arr)
        return torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        labels = self._sk.fit_predict(arr)
        centers = self._sk.cluster_centers_
        inertia = float(self._sk.inertia_)

        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        centers_t = torch.from_numpy(np.asarray(centers, dtype=np.float32))
        # move centers to same device as input
        try:
            centers_t = centers_t.to(x.device)
        except Exception:
            pass

        counts = torch.bincount(labels_t, minlength=self.n_clusters).tolist()
        cluster_sizes = {i: int(c) for i, c in enumerate(counts) if c > 0}

        return ClusteringResult(
            labels=labels_t,
            n_clusters_found=len(cluster_sizes),
            centroids=centers_t,
            objective=inertia,
            cluster_sizes=cluster_sizes,
            metadata={
                "algorithm": "sklearn_kmeans",
                "n_clusters": self.config.n_clusters,
                "max_iter": self.config.max_iter,
                "tol": self.config.tol,
                "seed": self.config.seed,
                "n_init": getattr(self._sk, "n_init", None),
                "init": getattr(self._sk, "init", None),
            },
        )
