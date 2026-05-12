from __future__ import annotations

from typing import cast

from .base import ClusteringAlgorithm, ClusteringConfig


class ClustererFactory:
	"""Factory that instantiates clusterer adapters by name using a config
	object. Mirrors the style of `FeatureExtractorFactory` (runtime imports
	and runtime config type checks).
	"""

	def create(self, name: str, config: ClusteringConfig) -> ClusteringAlgorithm:
		"""Create an instance of a supported clusterer by name.

		The caller must pass a `ClusteringConfig` instance of the correct
		concrete subclass expected by the adapter (the factory validates
		this at runtime and raises `TypeError` on mismatch).
		"""
		if not isinstance(name, str):
			raise TypeError("name must be a string")
		key = name.strip().lower()

		match key:
			case "kmeans":
				from .kmeans import SklearnKMeansAdapter, KMeansConfig

				if not isinstance(config, KMeansConfig):
					raise TypeError("config must be a KMeansConfig for clusterer 'kmeans'")
				cfg = cast(KMeansConfig, config)
				return SklearnKMeansAdapter(cfg)

			case "agglomerative":
				from .agglomerativeClustering import SklearnAgglomerativeAdapter, AgglomerativeConfig

				if not isinstance(config, AgglomerativeConfig):
					raise TypeError("config must be an AgglomerativeConfig for clusterer 'agglomerative'")
				cfg = cast(AgglomerativeConfig, config)
				return SklearnAgglomerativeAdapter(cfg)

			case "spectral":
				from .spectralClustering import SklearnSpectralClusteringAdapter, SpectralClusteringConfig

				if not isinstance(config, SpectralClusteringConfig):
					raise TypeError("config must be a SpectralClusteringConfig for clusterer 'spectral'")
				cfg = cast(SpectralClusteringConfig, config)
				return SklearnSpectralClusteringAdapter(cfg)

			case "dbscan":
				from .dbscan import SklearnDBSCANAdapter, DBSCANConfig

				if not isinstance(config, DBSCANConfig):
					raise TypeError("config must be a DBSCANConfig for clusterer 'dbscan'")
				cfg = cast(DBSCANConfig, config)
				return SklearnDBSCANAdapter(cfg)

			case "optics":
				from .optics import SklearnOpticsAdapter, OpticsConfig

				if not isinstance(config, OpticsConfig):
					raise TypeError("config must be an OpticsConfig for clusterer 'optics'")
				cfg = cast(OpticsConfig, config)
				return SklearnOpticsAdapter(cfg)

			case "affinity_propagation" | "affinity":
				from .affinityPropagation import SklearnAffinityPropagationAdapter, AffinityPropagationConfig

				if not isinstance(config, AffinityPropagationConfig):
					raise TypeError("config must be an AffinityPropagationConfig for clusterer 'affinity_propagation'")
				cfg = cast(AffinityPropagationConfig, config)
				return SklearnAffinityPropagationAdapter(cfg)

			case "gmm" | "gaussian_mixture":
				from .gaussianMixture import SklearnGMMAdapter, GMMConfig

				if not isinstance(config, GMMConfig):
					raise TypeError("config must be a GMMConfig for clusterer 'gmm'")
				cfg = cast(GMMConfig, config)
				return SklearnGMMAdapter(cfg)

			case "hdbscan":
				from .hdbscan import HDBSCANAdapter, HDBSCANConfig

				if not isinstance(config, HDBSCANConfig):
					raise TypeError("config must be a HDBSCANConfig for clusterer 'hdbscan'")
				cfg = cast(HDBSCANConfig, config)
				return HDBSCANAdapter(cfg)

			case _:
				raise ValueError(f"Unknown clusterer: {name}")
