from __future__ import annotations

from typing import Any, Optional, cast, Literal, Sequence

from src.clustering.gaussianMixture import GMMConfig
from .config_section_reader import ConfigSectionReader


class GaussianMixtureConfigReader(ConfigSectionReader[GMMConfig]):
    """Reads the `gaussianMixture` section and returns a `GMMConfig`.

    Expects the raw mapping (the full YAML root) and validates the
    `gaussianMixture` mapping inside it.
    """

    def read_section(self, raw: dict[str, Any]) -> GMMConfig:
        gm_cfg = self.require_mapping(raw, "gaussianMixture")
        # Parse n_components and optional range
        n_components_range: tuple[int, int] | None = None
        if "n_components_range" in gm_cfg:
            n_components_range_raw: Any = self.require_value(gm_cfg, "n_components_range")
            if not isinstance(n_components_range_raw, (list, tuple)):
                raise ValueError("gaussianMixture.n_components_range must be a list or tuple with two integers")
            n_components_range_vals = cast(Sequence[Any], n_components_range_raw)
            if len(n_components_range_vals) != 2:
                raise ValueError("gaussianMixture.n_components_range must have exactly two values")
            n_comp_start = int(n_components_range_vals[0])
            n_comp_end = int(n_components_range_vals[1])
            if n_comp_start > n_comp_end:
                raise ValueError("gaussianMixture.n_components_range start must be <= end")
            n_components_range = (n_comp_start, n_comp_end)
            n_components = n_comp_start
        else:
            n_components = int(self.require_value(gm_cfg, "n_components"))

        tol = float(self.require_value(gm_cfg, "tol"))
        reg_covar = float(self.require_value(gm_cfg, "reg_covar"))
        max_iter = int(self.require_value(gm_cfg, "max_iter"))
        n_init = int(self.require_value(gm_cfg, "n_init"))
        init_params = str(self.require_value(gm_cfg, "init_params"))

        allowed_init = {"kmeans", "k-means++", "random", "random_from_data"}
        if init_params not in allowed_init:
            raise ValueError(f"Invalid init_params for gaussianMixture: {init_params}")

        # Parse random_state and optional range
        random_state_range: tuple[int, int] | None = None
        if "random_state_range" in gm_cfg:
            random_state_range_raw: Any = self.require_value(gm_cfg, "random_state_range")
            if not isinstance(random_state_range_raw, (list, tuple)):
                raise ValueError("gaussianMixture.random_state_range must be a list or tuple with two integers")
            random_state_range_vals = cast(Sequence[Any], random_state_range_raw)
            if len(random_state_range_vals) != 2:
                raise ValueError("gaussianMixture.random_state_range must have exactly two values")
            rs_start = int(random_state_range_vals[0])
            rs_end = int(random_state_range_vals[1])
            if rs_start > rs_end:
                raise ValueError("gaussianMixture.random_state_range start must be <= end")
            random_state_range = (rs_start, rs_end)
            random_state = rs_start
        else:
            random_state = int(self.require_value(gm_cfg, "random_state"))

        covariance_type = str(self.require_value(gm_cfg, "covariance_type"))
        allowed_cov = {"full", "tied", "diag", "spherical"}
        if covariance_type not in allowed_cov:
            raise ValueError(f"Invalid covariance_type for gaussianMixture: {covariance_type}")

        init_params_cast = cast(Literal['kmeans', 'k-means++', 'random', 'random_from_data'], init_params)
        covariance_type_cast = cast(Literal['full', 'tied', 'diag', 'spherical'], covariance_type)

        # Parse n_trials
        n_trials = int(self.optional_value(gm_cfg, "n_trials", 50))

        return GMMConfig(
            n_components=n_components,
            n_components_range=n_components_range,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params_cast,
            random_state=random_state,
            random_state_range=random_state_range,
            covariance_type=covariance_type_cast,
            n_trials=n_trials,
        )
