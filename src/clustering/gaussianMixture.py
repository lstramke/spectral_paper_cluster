from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from sklearn.mixture import GaussianMixture as SKGaussianMixture

from .base import ClusteringAlgorithm, ClusteringResult


@dataclass(slots=True)
class GMMConfig:
    n_components: int
    tol: float
    reg_covar: float
    max_iter: int
    n_init: int
    init_params: Literal['kmeans', 'k-means++', 'random', 'random_from_data']
    random_state: int
    covariance_type: Literal['full', 'tied', 'diag', 'spherical']


class SklearnGMMAdapter(ClusteringAlgorithm):
    """Adapter around sklearn.mixture.GaussianMixture implementing the
    project's `ClusteringAlgorithm`/`ClusteringResult` contract.
    """

    def __init__(self, config: GMMConfig, **sk_kwargs) -> None:
        self.config = config
        self._sk = SKGaussianMixture(
            n_components=config.n_components,
            tol=config.tol,
            reg_covar=config.reg_covar,
            max_iter=config.max_iter,
            n_init=config.n_init,
            init_params=config.init_params,
            random_state=config.random_state,
            covariance_type=config.covariance_type,
            **sk_kwargs,
        )

    def fit(self, x: Tensor) -> ClusteringAlgorithm:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        self._sk.fit(arr)
        return self

    def predict(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()
        labels = self._sk.predict(arr)
        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        try:
            labels_t = labels_t.to(x.device)
        except Exception:
            pass
        return labels_t

    def fit_predict(self, x: Tensor) -> ClusteringResult:
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor")
        arr = x.detach().cpu().numpy()

        labels = self._sk.fit_predict(arr)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

        # number of components actually occupied
        n_clusters = int(np.sum(counts > 0))

        labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        try:
            labels_t = labels_t.to(torch.device(x.device))
        except Exception:
            pass

        # component means provided by the GMM
        means = getattr(self._sk, "means_", None)
        centers_t = None
        if means is not None:
            centers_t = torch.from_numpy(np.asarray(means, dtype=np.float32))
            try:
                centers_t = centers_t.to(x.device)
            except Exception:
                pass

        objective = None
        if hasattr(self._sk, "lower_bound_"):
            try:
                objective = float(self._sk.lower_bound_)
            except Exception:
                objective = None

        metadata = {
            "algorithm": "sklearn_gmm",
            "n_components": self.config.n_components,
            "covariance_type": self.config.covariance_type,
            "tol": self.config.tol,
            "reg_covar": self.config.reg_covar,
            "max_iter": self.config.max_iter,
            "n_init": self.config.n_init,
            "init_params": self.config.init_params,
            "random_state": self.config.random_state,
        }

        # AIC/BIC if available
        try:
            metadata["aic"] = float(self._sk.aic(arr))
            metadata["bic"] = float(self._sk.bic(arr))
        except Exception:
            pass

        return ClusteringResult(
            labels=labels_t,
            n_clusters_found=n_clusters,
            centroids=centers_t,
            objective=objective,
            cluster_sizes=cluster_sizes,
            metadata=metadata,
        )
