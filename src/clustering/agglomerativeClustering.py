from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import AgglomerativeClustering as SKLearnAgglomerative

from app_types.optimization_field import OptimizationField

from .base import ClusteringAlgorithm, ClusteringResult, ClusteringConfig


@dataclass(slots=True)
class AgglomerativeConfig(ClusteringConfig):
    distance_threshold: float | None
    distance_threshold_range: tuple[float, float] | None
    n_clusters: int | None
    metric: str
    linkage: Literal['ward', 'complete', 'average', 'single']
    compute_full_tree: bool
    n_trials: int

    def get_n_trials(self) -> int:
        return self.n_trials

    def get_optimization_fields(self) -> list[OptimizationField]:
        fields: list[OptimizationField] = []

        distance_threshold_min, distance_threshold_max = self.distance_threshold_range or (self.distance_threshold, self.distance_threshold)
        assert distance_threshold_min is not None
        assert distance_threshold_max is not None
        fields.append(
            OptimizationField[float](
                name="distance_threshold",
                min_value=distance_threshold_min,
                max_value=distance_threshold_max,
                value_type=float
            )
        )
        return fields


class SklearnAgglomerativeAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.AgglomerativeClustering implementing the
    project's `ClusteringAlgorithm`/`ClusteringResult` contract.
    """

    def __init__(self, config: AgglomerativeConfig) -> None:
        self.config = config
        self._sk = SKLearnAgglomerative(
            n_clusters=config.n_clusters,
            metric=config.metric,
            linkage=config.linkage,
            distance_threshold=config.distance_threshold,
            compute_full_tree=config.compute_full_tree,
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
