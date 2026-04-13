from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import DBSCAN as SKLearnDBSCAN

from .base import ClusteringAlgorithm, ClusteringResult


@dataclass(slots=True)
class DBSCANConfig:
    eps: float
    min_samples: int
    metric: str
    leaf_size: int
    p: Optional[int]
    n_jobs: Optional[int]


class SklearnDBSCANAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.DBSCAN implementing the project's
    `ClusteringAlgorithm`/`ClusteringResult` contract.
    """

    def __init__(self, config: DBSCANConfig, **sk_kwargs) -> None:
        self.config = config
        self._sk = SKLearnDBSCAN(
            eps=config.eps,
            min_samples=config.min_samples,
            metric=config.metric,
            leaf_size=config.leaf_size,
            p=config.p,
            n_jobs=config.n_jobs,
            **sk_kwargs,
        )

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        self._sk.fit(arr)
        return self

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        labels = self._sk.fit_predict(arr)

        # labels contain -1 for noise. Compute unique labels and counts.
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

        # number of clusters (exclude noise label -1 if present)
        n_clusters = int(np.sum(unique != -1))

        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        # compute centroids for each non-noise cluster
        centroids_list = []
        cluster_ids = [int(u) for u in unique if u != -1]
        if cluster_ids:
            for cid in cluster_ids:
                mask = labels == cid
                centroid = arr[mask].mean(axis=0)
                centroids_list.append(centroid)
            centers_t = torch.from_numpy(np.asarray(centroids_list, dtype=np.float32))
            try:
                centers_t = centers_t.to(x.device)
            except Exception:
                pass
        else:
            centers_t = None

        return ClusteringResult(
            labels=labels_t,
            n_clusters_found=n_clusters,
            centroids=centers_t,
            objective=None,
            cluster_sizes=cluster_sizes,
            metadata={
                "algorithm": "sklearn_dbscan",
                "eps": self.config.eps,
                "min_samples": self.config.min_samples,
                "metric": self.config.metric,
            },
        )
