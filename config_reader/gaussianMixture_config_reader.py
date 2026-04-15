from __future__ import annotations

from typing import Any, Optional, cast, Literal

from src.clustering.gaussianMixture import GMMConfig
from .config_section_reader import ConfigSectionReader


class GaussianMixtureConfigReader(ConfigSectionReader[GMMConfig]):
    """Reads the `gaussianMixture` section and returns a `GMMConfig`.

    Expects the raw mapping (the full YAML root) and validates the
    `gaussianMixture` mapping inside it.
    """

    def read_section(self, raw: dict[str, Any]) -> GMMConfig:
        gm_cfg = self.require_mapping(raw, "gaussianMixture")

        n_components = int(self.require_value(gm_cfg, "n_components"))
        tol = float(self.require_value(gm_cfg, "tol"))
        reg_covar = float(self.require_value(gm_cfg, "reg_covar"))
        max_iter = int(self.require_value(gm_cfg, "max_iter"))
        n_init = int(self.require_value(gm_cfg, "n_init"))
        init_params = str(self.require_value(gm_cfg, "init_params"))

        allowed_init = {"kmeans", "k-means++", "random", "random_from_data"}
        if init_params not in allowed_init:
            raise ValueError(f"Invalid init_params for gaussianMixture: {init_params}")

        random_state = int(self.require_value(gm_cfg, "random_state"))

        covariance_type = str(self.require_value(gm_cfg, "covariance_type"))
        allowed_cov = {"full", "tied", "diag", "spherical"}
        if covariance_type not in allowed_cov:
            raise ValueError(f"Invalid covariance_type for gaussianMixture: {covariance_type}")

        init_params_cast = cast(Literal['kmeans', 'k-means++', 'random', 'random_from_data'], init_params)
        covariance_type_cast = cast(Literal['full', 'tied', 'diag', 'spherical'], covariance_type)

        return GMMConfig(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params_cast,
            random_state=random_state,
            covariance_type=covariance_type_cast,
        )
