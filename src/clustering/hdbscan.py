from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

import hdbscan

from .base import ClusteringAlgorithm, ClusteringResult


@dataclass(slots=True)
class HDBSCANConfig:
    min_cluster_size: int
    min_samples: Optional[int]
    metric: str
    cluster_selection_method: str


class HDBSCANAdapter(ClusteringAlgorithm):
    """Adapter around the `hdbscan` package implementing the project's
    `ClusteringAlgorithm`/`ClusteringResult` contract.
    """

    def __init__(self, config: HDBSCANConfig, **hdbscan_kwargs) -> None:
        self.config = config
        

        self._model = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric=self.config.metric,
            cluster_selection_method=self.config.cluster_selection_method,
            **hdbscan_kwargs,
        )

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        """L2-normalize rows of `arr`. Zeros are left as zeros (avoids div-by-zero)."""
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        return arr / norms

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        # If using euclidean on TF-IDF/LSA vectors, L2-normalize so Euclidean
        # distance behaves like cosine similarity.
        if str(self.config.metric).lower() in ("euclidean", "l2"):
            arr = self._l2_normalize(arr)
        self._model.fit(arr)
        return self

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        # fit if not already fitted
        try:
            labels = self._model.labels_
        except AttributeError:
            if str(self.config.metric).lower() in ("euclidean", "l2"):
                arr = self._l2_normalize(arr)
            self._model.fit(arr)
            labels = self._model.labels_

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

        # optional soft membership probabilities provided by hdbscan
        probabilities_available = hasattr(self._model, "probabilities_")

        metadata = {
            "algorithm": "hdbscan",
            "min_cluster_size": self.config.min_cluster_size,
            "min_samples": self.config.min_samples,
            "metric": self.config.metric,
            "cluster_selection_method": self.config.cluster_selection_method,
            "probabilities_available": probabilities_available,
        }

        return ClusteringResult(
            labels=labels_t,
            n_clusters_found=n_clusters,
            centroids=centers_t,
            objective=None,
            cluster_sizes=cluster_sizes,
            metadata=metadata,
        )
