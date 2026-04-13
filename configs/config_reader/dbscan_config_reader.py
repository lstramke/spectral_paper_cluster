from __future__ import annotations

from typing import Any, cast

from src.clustering.dbscan import DBSCANConfig
from .config_section_reader import ConfigSectionReader


class DbscanConfigReader(ConfigSectionReader[DBSCANConfig]):
	"""Reads the `dbscan` section and returns a `DBSCANConfig`.

	Expects the raw mapping (the full YAML root) and validates the
	`dbscan` mapping inside it.
	"""

	def read_section(self, raw: dict[str, Any]) -> DBSCANConfig:
		dbscan_cfg = self.require_mapping(raw, "dbscan")

		eps = float(self.require_value(dbscan_cfg, "eps"))
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

		return DBSCANConfig(
			eps=eps,
			min_samples=min_samples,
			metric=metric,
			leaf_size=leaf_size,
			p=p,
			n_jobs=n_jobs,
		)
