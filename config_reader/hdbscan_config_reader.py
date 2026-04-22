from __future__ import annotations

from typing import Any, cast, Sequence

from src.clustering.hdbscan import HDBSCANConfig
from .config_section_reader import ConfigSectionReader


class HdbscanConfigReader(ConfigSectionReader[HDBSCANConfig]):
	"""Reads the `hdbscan` section and returns a `HDBSCANConfig`.

	Expects the raw mapping (the full YAML root) and validates the
	`hdbscan` mapping inside it.
	"""

	def read_section(self, raw: dict[str, Any]) -> HDBSCANConfig:
		hdb_cfg = self.require_mapping(raw, "hdbscan")

		min_cluster_size_range: tuple[int, int] | None = None
		if "min_cluster_size_range" in hdb_cfg:
			mcs_range_raw: Any = self.require_value(hdb_cfg, "min_cluster_size_range")
			if not isinstance(mcs_range_raw, (list, tuple)):
				raise ValueError("hdbscan.min_cluster_size_range must be a list or tuple with two integers")
			mcs_range_vals = cast(Sequence[Any], mcs_range_raw)
			if len(mcs_range_vals) != 2:
				raise ValueError("hdbscan.min_cluster_size_range must have exactly two values")
			mcs_start = int(mcs_range_vals[0])
			mcs_end = int(mcs_range_vals[1])
			if mcs_start > mcs_end:
				raise ValueError("hdbscan.min_cluster_size_range start must be <= end")
			min_cluster_size_range = (mcs_start, mcs_end)
			min_cluster_size = mcs_start
		else:
			min_cluster_size = int(self.require_value(hdb_cfg, "min_cluster_size"))

		min_samples_range: tuple[int, int] | None = None
		if "min_samples_range" in hdb_cfg:
			ms_range_raw: Any = self.require_value(hdb_cfg, "min_samples_range")
			if not isinstance(ms_range_raw, (list, tuple)):
				raise ValueError("hdbscan.min_samples_range must be a list or tuple with two integers")
			ms_range_vals = cast(Sequence[Any], ms_range_raw)
			if len(ms_range_vals) != 2:
				raise ValueError("hdbscan.min_samples_range must have exactly two values")
			ms_start = int(ms_range_vals[0])
			ms_end = int(ms_range_vals[1])
			if ms_start > ms_end:
				raise ValueError("hdbscan.min_samples_range start must be <= end")
			min_samples_range = (ms_start, ms_end)
			min_samples = ms_start
		else:
			min_samples: int | None = None
			if "min_samples" in hdb_cfg:
				raw_ms = hdb_cfg["min_samples"]
				if raw_ms is not None:
					min_samples = int(raw_ms)

		metric = str(self.require_value(hdb_cfg, "metric"))

		cluster_selection_method = str(self.optional_value(hdb_cfg, "cluster_selection_method", "eom"))

		n_trials = int(self.require_value(hdb_cfg, "n_trials"))

		return HDBSCANConfig(
			min_cluster_size=min_cluster_size,
			min_cluster_size_range=min_cluster_size_range,
			min_samples=min_samples,
			min_samples_range=min_samples_range,
			metric=metric,
			cluster_selection_method=cluster_selection_method,
			n_trials=n_trials,
		)
