from __future__ import annotations

from typing import Any, cast, Sequence

from src.clustering.affinityPropagation import AffinityPropagationConfig
from .config_section_reader import ConfigSectionReader


class AffinityPropagationConfigReader(ConfigSectionReader[AffinityPropagationConfig]):
    """
    Reads the `affinityPropagation` section and returns an `AffinityPropagationConfig`
    Raises ValueError on missing/invalid values.
    """

    def read_section(self, raw: dict[str, Any]) -> AffinityPropagationConfig:
        ap = self.require_mapping(raw, "affinityPropagation")

        # Parse damping and damping_range
        damping_range: tuple[float, float] | None = None
        if "damping_range" in ap:
            damping_range_raw: Any = self.require_value(ap, "damping_range")
            if not isinstance(damping_range_raw, (list, tuple)):
                raise ValueError("affinityPropagation.damping_range must be a list or tuple with two floats")
            damping_range_values = cast(Sequence[Any], damping_range_raw)
            if len(damping_range_values) != 2:
                raise ValueError("affinityPropagation.damping_range must have exactly two values")
            damping_start = float(damping_range_values[0])
            damping_end = float(damping_range_values[1])
            if damping_start > damping_end:
                raise ValueError("affinityPropagation.damping_range start must be <= end")
            damping_range = (damping_start, damping_end)
            damping = damping_start
        else:
            damping = float(self.require_value(ap, "damping"))

        # Parse random_state and random_state_range
        random_state_range: tuple[int, int] | None = None
        if "random_state_range" in ap:
            random_state_range_raw: Any = self.require_value(ap, "random_state_range")
            if not isinstance(random_state_range_raw, (list, tuple)):
                raise ValueError("affinityPropagation.random_state_range must be a list or tuple with two integers")
            random_state_range_values = cast(Sequence[Any], random_state_range_raw)
            if len(random_state_range_values) != 2:
                raise ValueError("affinityPropagation.random_state_range must have exactly two values")
            random_state_start = int(random_state_range_values[0])
            random_state_end = int(random_state_range_values[1])
            if random_state_start > random_state_end:
                raise ValueError("affinityPropagation.random_state_range start must be <= end")
            random_state_range = (random_state_start, random_state_end)
            random_state = random_state_start
        else:
            random_state = int(self.require_value(ap, "random_state"))

        max_iter = int(self.require_value(ap, "max_iter"))
        convergence_iter = int(self.require_value(ap, "convergence_iter"))

        affinity = str(self.optional_value(ap, "affinity", "euclidean"))
        if affinity not in ("euclidean", "precomputed"):
            raise ValueError(f"Invalid affinity: {affinity}")

        normalize = bool(self.require_value(ap, "normalize"))

        n_trials = int(self.require_value(ap, "n_trials"))

        return AffinityPropagationConfig(
            damping=damping,
            damping_range=damping_range,
            random_state=random_state,
            random_state_range=random_state_range,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            affinity=affinity,
            normalize=normalize,
            n_trials=n_trials,
        )
