from __future__ import annotations

from typing import Any

from src.clustering.spectralClustering import SpectralClusteringConfig
from .config_section_reader import ConfigSectionReader


class SpectralConfigReader(ConfigSectionReader[SpectralClusteringConfig]):
    """Reads the `spectral` section and returns a `SpectralClusteringConfig`.

    Expects the raw mapping (the full YAML root) and validates the
    `spectral` mapping inside it.
    """

    def read_section(self, raw: dict[str, Any]) -> SpectralClusteringConfig:
        spectral_cfg = self.require_mapping(raw, "spectral")

        n_clusters = int(self.require_value(spectral_cfg, "n_clusters"))
        affinity = str(self.require_value(spectral_cfg, "affinity"))
        eigen_solver = str(self.require_value(spectral_cfg, "eigen_solver"))
        assign_labels = str(self.require_value(spectral_cfg, "assign_labels"))
        n_init = int(self.require_value(spectral_cfg, "n_init"))
        gamma = float(self.require_value(spectral_cfg, "gamma"))
        n_neighbors = int(self.require_value(spectral_cfg, "n_neighbors"))

        random_state = int(self.require_value(spectral_cfg, "random_state"))
        n_jobs = int(self.require_value(spectral_cfg, "n_jobs"))

        return SpectralClusteringConfig(
            n_clusters=n_clusters,
            affinity=affinity,
            eigen_solver=eigen_solver,
            assign_labels=assign_labels,
            n_init=n_init,
            gamma=gamma,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )
