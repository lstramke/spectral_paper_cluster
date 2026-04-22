from __future__ import annotations

from typing import Any, cast, Sequence

from src.clustering.dbscan import DBSCANConfig
from .config_section_reader import ConfigSectionReader


class DbscanConfigReader(ConfigSectionReader[DBSCANConfig]):
	"""Reads the `dbscan` section and returns a `DBSCANConfig`.

	Expects the raw mapping (the full YAML root) and validates the
	`dbscan` mapping inside it.
	"""

	def read_section(self, raw: dict[str, Any]) -> DBSCANConfig:
		dbscan_cfg = self.require_mapping(raw, "dbscan")

		eps_range: tuple[float, float] | None = None
		if "eps_range" in dbscan_cfg:
			eps_range_raw: Any = self.require_value(dbscan_cfg, "eps_range")
			if not isinstance(eps_range_raw, (list, tuple)):
				raise ValueError("dbscan.eps_range must be a list or tuple with two floats")
			eps_range_values = cast(Sequence[Any], eps_range_raw)
			if len(eps_range_values) != 2:
				raise ValueError("dbscan.eps_range must have exactly two values")
			eps_start = float(eps_range_values[0])
			eps_end = float(eps_range_values[1])
			if eps_start > eps_end:
				raise ValueError("dbscan.eps_range start must be <= end")
			eps_range = (eps_start, eps_end)
			eps = eps_start
		else:
			eps = float(self.require_value(dbscan_cfg, "eps"))

		min_samples_range: tuple[int, int] | None = None
		if "min_samples_range" in dbscan_cfg:
			min_samples_range_raw: Any = self.require_value(dbscan_cfg, "min_samples_range")
			if not isinstance(min_samples_range_raw, (list, tuple)):
				raise ValueError("dbscan.min_samples_range must be a list or tuple with two integers")
			min_samples_range_values = cast(Sequence[Any], min_samples_range_raw)
			if len(min_samples_range_values) != 2:
				raise ValueError("dbscan.min_samples_range must have exactly two values")
			min_samples_start = int(min_samples_range_values[0])
			min_samples_end = int(min_samples_range_values[1])
			if min_samples_start > min_samples_end:
				raise ValueError("dbscan.min_samples_range start must be <= end")
			min_samples_range = (min_samples_start, min_samples_end)
			min_samples = min_samples_start
		else:
			min_samples = int(self.require_value(dbscan_cfg, "min_samples"))

		metric = str(self.require_value(dbscan_cfg, "metric"))
		leaf_size = int(self.require_value(dbscan_cfg, "leaf_size"))

		p: int | None = None
		if "p" in dbscan_cfg:
			raw_p = dbscan_cfg["p"]
			if raw_p is not None:
				p = int(raw_p)

		n_jobs: int | None = None
		if "n_jobs" in dbscan_cfg:
			raw_n = dbscan_cfg["n_jobs"]
			if raw_n is not None:
				n_jobs = int(raw_n)

		n_trials = int(self.require_value(dbscan_cfg, "n_trials"))

		return DBSCANConfig(
			eps=eps,
			eps_range=eps_range,
			min_samples=min_samples,
			min_samples_range=min_samples_range,
			metric=metric,
			leaf_size=leaf_size,
			p=p,
			n_jobs=n_jobs,
			n_trials=n_trials,
		)
