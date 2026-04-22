from __future__ import annotations

from typing import Any, Sequence, cast

from src.clustering.kmeans import KMeansConfig
from .config_section_reader import ConfigSectionReader


class KMeansConfigReader(ConfigSectionReader[KMeansConfig]):
	"""Reads the `kmeans` section and returns a `KMeansConfig`.

	Expects the raw mapping (the full YAML root) and validates the
	`kmeans` mapping inside it.
	"""

	def read_section(self, raw: dict[str, Any]) -> KMeansConfig:
		kmeans_cfg = self.require_mapping(raw, "kmeans")

		max_iter = int(self.require_value(kmeans_cfg, "max_iter"))
		tol = float(self.require_value(kmeans_cfg, "tol"))
		n_trials: int = self.require_value(kmeans_cfg, "n_trials")

		seed_range: tuple[int, int] | None = None
		if "seed_range" in kmeans_cfg:
			seed_range_raw: Any = self.require_value(kmeans_cfg, "seed_range")
			if not isinstance(seed_range_raw, (list, tuple)):
				raise ValueError("kmeans.seed_range must be a list or tuple with two integers")
			seed_range_values = cast(Sequence[Any], seed_range_raw)
			if len(seed_range_values) != 2:
				raise ValueError("kmeans.seed_range must have exactly two values")
			seed_start = int(seed_range_values[0])
			seed_end = int(seed_range_values[1])
			if seed_start > seed_end:
				raise ValueError("kmeans.seed_range start must be <= end")
			seed_range = (seed_start, seed_end)
			seed = seed_start
		else:
			seed = int(self.require_value(kmeans_cfg, "seed"))

		cluster_range: tuple[int, int] | None = None
		if "cluster_range" in kmeans_cfg:
			cluster_range_raw: Any = self.require_value(kmeans_cfg, "cluster_range")
			if not isinstance(cluster_range_raw, (list, tuple)):
				raise ValueError("kmeans.cluster_range must be a list or tuple with two integers")
			cluster_range_values = cast(Sequence[Any], cluster_range_raw)
			if len(cluster_range_values) != 2:
				raise ValueError("kmeans.cluster_range must have exactly two values")
			cluster_start = int(cluster_range_values[0])
			cluster_end = int(cluster_range_values[1])
			if cluster_start > cluster_end:
				raise ValueError("kmeans.cluster_range start must be <= end")
			cluster_range = (cluster_start, cluster_end)
			n_clusters = cluster_start
		else:
			n_clusters = int(self.require_value(kmeans_cfg, "n_clusters"))
			
		return KMeansConfig(
			n_clusters=n_clusters,
			cluster_range=cluster_range,
			max_iter=max_iter,
			tol=tol,
			seed=seed,
			seed_range=seed_range,
			n_trials=n_trials
		)