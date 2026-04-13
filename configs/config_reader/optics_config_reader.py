from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.clustering.optics import OpticsConfig
from .config_section_reader import ConfigSectionReader


class OpticsConfigReader(ConfigSectionReader[OpticsConfig]):
    """Reads the `optics` section and returns an `OpticsConfig`.

    Follows the same style as `KMeansConfigReader`.
    """

    def read_section(self, raw: dict[str, Any]) -> OpticsConfig:
        optics_cfg = self.require_mapping(raw, "optics")

        min_samples = int(self.require_value(optics_cfg, "min_samples"))
        metric = str(self.require_value(optics_cfg, "metric"))
        xi = float(self.require_value(optics_cfg, "xi"))

        # optional values with sensible defaults matching OpticsConfig
        cluster_method = str(optics_cfg.get("cluster_method", "xi"))

        n_jobs: Optional[int] = None
        if "n_jobs" in optics_cfg:
            n_jobs_val = self.require_value(optics_cfg, "n_jobs")
            n_jobs = None if n_jobs_val is None else int(n_jobs_val)

        if min_samples <= 0:
            raise ValueError("optics.min_samples must be > 0")
        if not (0.0 <= xi <= 1.0):
            raise ValueError("optics.xi must be in [0, 1]")

        return OpticsConfig(
            min_samples=min_samples,
            metric=metric,
            cluster_method=cluster_method,
            xi=xi,
            n_jobs=n_jobs,
        )
