from __future__ import annotations

from typing import Any

from src.clustering.affinityPropagation import AffinityPropagationConfig
from .config_section_reader import ConfigSectionReader


class AffinityPropagationConfigReader(ConfigSectionReader[AffinityPropagationConfig]):
    """
    Reads the `affinityPropagation` section and returns an `AffinityPropagationConfig`
    Raises ValueError on missing/invalid values.
    """

    def read_section(self, raw: dict[str, Any]) -> AffinityPropagationConfig:
        ap = self.require_mapping(raw, "affinityPropagation")

        try:
            damping = float(ap["damping"])
        except KeyError:
            raise ValueError("affinityPropagation.damping is required")

        try:
            max_iter = int(ap["max_iter"])
        except KeyError:
            raise ValueError("affinityPropagation.max_iter is required")

        try:
            convergence_iter = int(ap["convergence_iter"])
        except KeyError:
            raise ValueError("affinityPropagation.convergence_iter is required")

        affinity = str(self.optional_value(ap, "affinity", "euclidean"))
        if affinity not in ("euclidean", "precomputed"):
            raise ValueError(f"Invalid affinity: {affinity}")

        if "random_state" not in ap or ap["random_state"] is None:
            raise ValueError("affinityPropagation.random_state is required and must be an integer")
        random_state = int(ap["random_state"])

        if "normalize" not in ap:
            raise ValueError("affinityPropagation.normalize is required and must be a boolean")
        normalize = bool(ap["normalize"])

        return AffinityPropagationConfig(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            affinity=affinity,
            random_state=random_state,
            normalize=normalize,
        )
