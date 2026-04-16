from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import SpectralClustering as SKLearnSpectralClustering
import warnings

from .base import ClusteringAlgorithm, ClusteringResult


@dataclass(slots=True)
class SpectralClusteringConfig:
    n_clusters: int
    affinity: str  # 'rbf', 'nearest_neighbors', 'precomputed'
    eigen_solver: str
    assign_labels: str
    n_init: int
    gamma: float
    n_neighbors: int
    random_state: int
    n_jobs: int


class SklearnSpectralClusteringAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.SpectralClustering that implements the
    project's `ClusteringAlgorithm`/`ClusteringResult` interface.

    The adapter supports passing a precomputed affinity matrix by setting
    `config.affinity == "precomputed"`. In that case, the input to
    `fit`/`fit_predict` must be a square affinity matrix (n_samples x n_samples),
    for example a cosine similarity matrix computed externally.
    """

    def __init__(self, config: SpectralClusteringConfig, **sk_kwargs) -> None:
        self.config = config
        params: dict[str, Any] = {
            "n_clusters": config.n_clusters,
            "affinity": config.affinity,
            "assign_labels": config.assign_labels,
            "n_init": config.n_init,
            "random_state": config.random_state,
        }
        if config.eigen_solver is not None:
            params["eigen_solver"] = config.eigen_solver
        if config.gamma is not None:
            params["gamma"] = config.gamma
        if config.n_neighbors is not None:
            params["n_neighbors"] = config.n_neighbors
        if config.n_jobs is not None:
            params["n_jobs"] = config.n_jobs

        params.update(sk_kwargs)
        self._sk = SKLearnSpectralClustering(**params)

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")

        arr = x.detach().cpu().numpy()

        # If affinity is precomputed, we expect a square affinity matrix
        if self.config.affinity == "precomputed":
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("For affinity='precomputed' x must be a square affinity matrix")
            # sklearn will treat input as affinity matrix when affinity='precomputed'
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=r".*affinity.*",
                )
                self._sk.fit(arr)
        else:
            # treat arr as feature matrix; sklearn will compute affinity internally
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=r".*affinity.*",
                )
                self._sk.fit(arr)

        return self

    def predict(self, x: Tensor) -> Tensor:
        # SpectralClustering in sklearn does not support out-of-sample prediction
        raise NotImplementedError("predict() is not supported for SpectralClustering")

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")

        arr = x.detach().cpu().numpy()

        if self.config.affinity == "precomputed":
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("For affinity='precomputed' x must be a square affinity matrix")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=r".*affinity.*",
                )
                labels = self._sk.fit_predict(arr)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=r".*affinity.*",
                )
                labels = self._sk.fit_predict(arr)

        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts) if c > 0}
        n_clusters = int(len(unique))

        return ClusteringResult(
            labels=labels_t,
            n_clusters_found=n_clusters,
            centroids=None,
            objective=None,
            cluster_sizes=cluster_sizes,
            metadata={
                "algorithm": "sklearn_spectral",
                "n_clusters": self.config.n_clusters,
                "affinity": self.config.affinity,
                "eigen_solver": self.config.eigen_solver,
                "assign_labels": self.config.assign_labels,
                "n_init": self.config.n_init,
                "gamma": self.config.gamma,
                "n_neighbors": self.config.n_neighbors,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
            },
        )
