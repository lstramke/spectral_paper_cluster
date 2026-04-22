from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import AgglomerativeClustering as SKLearnAgglomerative

from .base import ClusteringAlgorithm, ClusteringResult


@dataclass(slots=True)
class AgglomerativeConfig:
    distance_threshold: float | None
    distance_threshold_range: tuple[float, float] | None
    n_clusters: int | None
    metric: str
    linkage: Literal['ward', 'complete', 'average', 'single']
    compute_full_tree: bool
    n_trials: int


class SklearnAgglomerativeAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.AgglomerativeClustering implementing the
    project's `ClusteringAlgorithm`/`ClusteringResult` contract.
    """

    def __init__(self, config: AgglomerativeConfig, **sk_kwargs) -> None:
        self.config = config
        self._sk = SKLearnAgglomerative(
            n_clusters=config.n_clusters,
            metric=config.metric,
            linkage=config.linkage,
            distance_threshold=config.distance_threshold,
            compute_full_tree=config.compute_full_tree,
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

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

        n_clusters = int(len(unique))

        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        # compute centroids for each cluster
        centroids_list = []
        cluster_ids = [int(u) for u in unique]
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
                "algorithm": "sklearn_agglomerative",
                "n_clusters": self.config.n_clusters,
                "metric": self.config.metric,
                "linkage": self.config.linkage,
            },
        )
