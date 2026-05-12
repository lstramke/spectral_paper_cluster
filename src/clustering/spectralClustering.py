from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import SpectralClustering as SKLearnSpectralClustering
import warnings

from app_types.optimization_field import OptimizationField

from .base import ClusteringAlgorithm, ClusteringResult, ClusteringConfig


@dataclass(slots=True)
class SpectralClusteringConfig(ClusteringConfig):
    n_clusters: int
    n_clusters_range: tuple[int, int] | None
    affinity: str  # 'rbf', 'nearest_neighbors', 'precomputed'
    eigen_solver: Literal["arpack", "lobpcg", "amg"]
    assign_labels: Literal["kmeans", "discretize", "cluster_qr"]
    n_init: int
    gamma: float
    n_neighbors: int
    n_neighbors_range: tuple[int, int] | None
    random_state: int
    random_state_range: tuple[int, int] | None
    n_jobs: int
    n_trials: int

    def get_n_trials(self) -> int:
        return self.n_trials

    def get_optimization_fields(self) -> list[OptimizationField]:
        fields: list[OptimizationField] = []
    
        min_clusters, max_clusters = self.n_clusters_range or (self.n_clusters, self.n_clusters)
        fields.append(
            OptimizationField[int](
                name="n_clusters",
                min_value=min_clusters,
                max_value=max_clusters,
                value_type=int,
            )
        )
    
        min_neighbors, max_neighbors = self.n_neighbors_range or (self.n_neighbors, self.n_neighbors)
        fields.append(
            OptimizationField[int](
                name="n_neighbors",
                min_value=min_neighbors,
                max_value=max_neighbors,
                value_type=int,
            )
        )
    
        random_start, random_end = self.random_state_range or (self.random_state, self.random_state)
        fields.append(
            OptimizationField[int](
                name="random_state",
                min_value=random_start,
                max_value=random_end,
                value_type=int
            )
        )
    
        return fields


class SklearnSpectralClusteringAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.cluster.SpectralClustering that implements the
    project's `ClusteringAlgorithm`/`ClusteringResult` interface.

    The adapter supports passing a precomputed affinity matrix by setting
    `config.affinity == "precomputed"`. In that case, the input to
    `fit`/`fit_predict` must be a square affinity matrix (n_samples x n_samples),
    for example a cosine similarity matrix computed externally.
    """

    def __init__(self, config: SpectralClusteringConfig) -> None:
        self.config = config
        # Pass parameters explicitly to avoid dict unpacking
        self._sk = SKLearnSpectralClustering(
            n_clusters=config.n_clusters,
            affinity=config.affinity,
            assign_labels=config.assign_labels,
            n_init=config.n_init,
            random_state=config.random_state,
            eigen_solver=config.eigen_solver,
            gamma=config.gamma,
            n_neighbors=config.n_neighbors,
            n_jobs=config.n_jobs,
        )

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
