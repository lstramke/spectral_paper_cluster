from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import AffinityPropagation as SKLearnAffinityPropagation
from sklearn.preprocessing import normalize as sk_normalize

from .base import ClusteringAlgorithm, ClusteringResult


@dataclass(slots=True)
class AffinityPropagationConfig:
    damping: float
    damping_range: tuple[float, float] | None
    random_state: int
    random_state_range: tuple[int, int] | None
    max_iter: int
    convergence_iter: int
    affinity: Literal["euclidean", "precomputed"]
    normalize: bool
    n_trials: int

class SklearnAffinityPropagationAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.AffinityPropagation implementing the
    project's `ClusteringAlgorithm`/`ClusteringResult` contract.
    """

    def __init__(self, config: AffinityPropagationConfig, **sk_kwargs) -> None:
        self.config = config
        self._sk = SKLearnAffinityPropagation(
            damping=config.damping,
            max_iter=config.max_iter,
            convergence_iter=config.convergence_iter,
            affinity=config.affinity,
            random_state=config.random_state,
            **sk_kwargs,
        )

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        
        if self.config.normalize:
            if self.config.affinity == "precomputed":
                raise ValueError("normalize=True is not supported when affinity='precomputed'")
            arr = sk_normalize(arr, norm="l2", axis=1, copy=False)

        self._sk.fit(arr)
        return self

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        if self.config.normalize:
            if self.config.affinity == "precomputed":
                raise ValueError("normalize=True is not supported when affinity='precomputed'")
            arr = sk_normalize(arr, norm="l2", axis=1, copy=False)

        labels = self._sk.fit_predict(arr)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
        n_clusters = int(unique.size)

        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        centers_t = None
        center_indices = getattr(self._sk, "cluster_centers_indices_", None)
        if center_indices is not None and len(center_indices) > 0:
            centers = arr[center_indices]
            centers_t = torch.from_numpy(np.asarray(centers, dtype=np.float32))
            try:
                centers_t = centers_t.to(x.device)
            except Exception:
                pass

        return ClusteringResult(
            labels=labels_t,
            n_clusters_found=n_clusters,
            centroids=centers_t,
            objective=None,
            cluster_sizes=cluster_sizes,
            metadata={
                "algorithm": "sklearn_affinity_propagation",
                "normalize": self.config.normalize,
                "damping": self.config.damping,
                "max_iter": self.config.max_iter,
                "convergence_iter": self.config.convergence_iter,
                "affinity": self.config.affinity,
            },
        )