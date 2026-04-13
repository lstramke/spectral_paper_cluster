from __future__ import annotations

from typing import Any, cast

from src.clustering.hdbscan import HDBSCANConfig
from .config_section_reader import ConfigSectionReader


class HdbscanConfigReader(ConfigSectionReader[HDBSCANConfig]):
	"""Reads the `hdbscan` section and returns a `HDBSCANConfig`.

	Expects the raw mapping (the full YAML root) and validates the
	`hdbscan` mapping inside it.
	"""

	def read_section(self, raw: dict[str, Any]) -> HDBSCANConfig:
		hdb_cfg = self.require_mapping(raw, "hdbscan")

		min_cluster_size = int(self.require_value(hdb_cfg, "min_cluster_size"))

		min_samples: int | None = None
		if "min_samples" in hdb_cfg:
			raw_ms = hdb_cfg["min_samples"]
			if raw_ms is not None:
				min_samples = int(raw_ms)

		metric = str(self.require_value(hdb_cfg, "metric"))

		cluster_selection_method = str(self.optional_value(hdb_cfg, "cluster_selection_method", "eom"))

		return HDBSCANConfig(
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			metric=metric,
			cluster_selection_method=cluster_selection_method,
		)
