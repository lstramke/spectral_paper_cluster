from __future__ import annotations

from typing import Any, Sequence, cast

from src.clustering.spectralClustering import SpectralClusteringConfig
from .config_section_reader import ConfigSectionReader


class SpectralConfigReader(ConfigSectionReader[SpectralClusteringConfig]):
    """Reads the `spectral` section and returns a `SpectralClusteringConfig`.

    Expects the raw mapping (the full YAML root) and validates the
    `spectral` mapping inside it.
    """

    def read_section(self, raw: dict[str, Any]) -> SpectralClusteringConfig:
        spectral_cfg = self.require_mapping(raw, "spectral")

        n_clusters_range: tuple[int, int] | None = None
        if "n_clusters_range" in spectral_cfg:
            nc_range_raw: Any = self.require_value(spectral_cfg, "n_clusters_range")
            if not isinstance(nc_range_raw, (list, tuple)):
                raise ValueError("spectral.n_clusters_range must be a list or tuple with two integers")
            nc_range_vals = cast(Sequence[Any], nc_range_raw)
            if len(nc_range_vals) != 2:
                raise ValueError("spectral.n_clusters_range must have exactly two values")
            nc_start = int(nc_range_vals[0])
            nc_end = int(nc_range_vals[1])
            if nc_start > nc_end:
                raise ValueError("spectral.n_clusters_range start must be <= end")
            n_clusters_range = (nc_start, nc_end)
            n_clusters = nc_start
        else:
            n_clusters = int(self.require_value(spectral_cfg, "n_clusters"))

        affinity = str(self.require_value(spectral_cfg, "affinity"))
        eigen_solver = str(self.require_value(spectral_cfg, "eigen_solver"))
        assign_labels = str(self.require_value(spectral_cfg, "assign_labels"))
        n_init = int(self.require_value(spectral_cfg, "n_init"))
        gamma = float(self.require_value(spectral_cfg, "gamma"))

        n_neighbors_range: tuple[int, int] | None = None
        if "n_neighbors_range" in spectral_cfg:
            nn_range_raw: Any = self.require_value(spectral_cfg, "n_neighbors_range")
            if not isinstance(nn_range_raw, (list, tuple)):
                raise ValueError("spectral.n_neighbors_range must be a list or tuple with two integers")
            nn_range_vals = cast(Sequence[Any], nn_range_raw)
            if len(nn_range_vals) != 2:
                raise ValueError("spectral.n_neighbors_range must have exactly two values")
            nn_start = int(nn_range_vals[0])
            nn_end = int(nn_range_vals[1])
            if nn_start > nn_end:
                raise ValueError("spectral.n_neighbors_range start must be <= end")
            n_neighbors_range = (nn_start, nn_end)
            n_neighbors = nn_start
        else:
            n_neighbors = int(self.require_value(spectral_cfg, "n_neighbors"))

        random_state_range: tuple[int, int] | None = None
        if "random_state_range" in spectral_cfg:
            rs_range_raw: Any = self.require_value(spectral_cfg, "random_state_range")
            if not isinstance(rs_range_raw, (list, tuple)):
                raise ValueError("spectral.random_state_range must be a list or tuple with two integers")
            rs_range_vals = cast(Sequence[Any], rs_range_raw)
            if len(rs_range_vals) != 2:
                raise ValueError("spectral.random_state_range must have exactly two values")
            rs_start = int(rs_range_vals[0])
            rs_end = int(rs_range_vals[1])
            if rs_start > rs_end:
                raise ValueError("spectral.random_state_range start must be <= end")
            random_state_range = (rs_start, rs_end)
            random_state = rs_start
        else:
            random_state = int(self.require_value(spectral_cfg, "random_state"))

        n_jobs = int(self.require_value(spectral_cfg, "n_jobs"))

        n_trials = int(self.require_value(spectral_cfg, "n_trials"))

        return SpectralClusteringConfig(
            n_clusters=n_clusters,
            n_clusters_range=n_clusters_range,
            affinity=affinity,
            eigen_solver=eigen_solver,
            assign_labels=assign_labels,
            n_init=n_init,
            gamma=gamma,
            n_neighbors=n_neighbors,
            n_neighbors_range=n_neighbors_range,
            random_state=random_state,
            random_state_range=random_state_range,
            n_jobs=n_jobs,
            n_trials=n_trials,
        )
