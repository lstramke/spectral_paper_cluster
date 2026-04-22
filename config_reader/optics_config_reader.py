from __future__ import annotations

from typing import Any, Optional, Sequence, cast

from src.clustering.optics import OpticsConfig
from .config_section_reader import ConfigSectionReader


class OpticsConfigReader(ConfigSectionReader[OpticsConfig]):
    """Reads the `optics` section and returns an `OpticsConfig`.

    Follows the same style as `KMeansConfigReader`.
    """

    def read_section(self, raw: dict[str, Any]) -> OpticsConfig:
        optics_cfg = self.require_mapping(raw, "optics")

        min_samples_range: tuple[int, int] | None = None
        if "min_samples_range" in optics_cfg:
            ms_range_raw: Any = self.require_value(optics_cfg, "min_samples_range")
            if not isinstance(ms_range_raw, (list, tuple)):
                raise ValueError("optics.min_samples_range must be a list or tuple with two integers")
            ms_range_vals = cast(Sequence[Any], ms_range_raw)
            if len(ms_range_vals) != 2:
                raise ValueError("optics.min_samples_range must have exactly two values")
            ms_start = int(ms_range_vals[0])
            ms_end = int(ms_range_vals[1])
            if ms_start > ms_end:
                raise ValueError("optics.min_samples_range start must be <= end")
            if ms_start <= 0:
                raise ValueError("optics.min_samples_range values must be > 0")
            min_samples_range = (ms_start, ms_end)
            min_samples = ms_start
        else:
            min_samples = int(self.require_value(optics_cfg, "min_samples"))
            if min_samples <= 0:
                raise ValueError("optics.min_samples must be > 0")

        metric = str(self.require_value(optics_cfg, "metric"))
        cluster_method = str(self.optional_value(optics_cfg, "cluster_method", "xi"))

        xi_range: tuple[float, float] | None = None
        if "xi_range" in optics_cfg:
            xi_range_raw: Any = self.require_value(optics_cfg, "xi_range")
            if not isinstance(xi_range_raw, (list, tuple)):
                raise ValueError("optics.xi_range must be a list or tuple with two floats")
            xi_range_vals = cast(Sequence[Any], xi_range_raw)
            if len(xi_range_vals) != 2:
                raise ValueError("optics.xi_range must have exactly two values")
            xi_start = float(xi_range_vals[0])
            xi_end = float(xi_range_vals[1])
            if xi_start > xi_end:
                raise ValueError("optics.xi_range start must be <= end")
            if not (0.0 <= xi_start <= 1.0 and 0.0 <= xi_end <= 1.0):
                raise ValueError("optics.xi_range values must be in [0, 1]")
            xi_range = (xi_start, xi_end)
            xi = xi_start
        else:
            xi = float(self.require_value(optics_cfg, "xi"))
            if not (0.0 <= xi <= 1.0):
                raise ValueError("optics.xi must be in [0, 1]")

        n_jobs: Optional[int] = None
        if "n_jobs" in optics_cfg:
            n_jobs_val = self.require_value(optics_cfg, "n_jobs")
            n_jobs = None if n_jobs_val is None else int(n_jobs_val)

        # Parse n_trials
        n_trials = int(self.require_value(optics_cfg, "n_trials"))

        return OpticsConfig(
            min_samples=min_samples,
            min_samples_range=min_samples_range,
            metric=metric,
            cluster_method=cluster_method,
            xi=xi,
            xi_range=xi_range,
            n_jobs=n_jobs,
            n_trials=n_trials,
        )
