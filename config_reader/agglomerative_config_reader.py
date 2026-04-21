from __future__ import annotations

from typing import Any, cast, Sequence

from src.clustering.agglomerativeClustering import AgglomerativeConfig
from .config_section_reader import ConfigSectionReader


class AgglomerativeConfigReader(ConfigSectionReader[AgglomerativeConfig]):
    """Reads the `agglomerative` section and returns an `AgglomerativeConfig`.

    Follows the same style as `DbscanConfigReader`: missing or invalid
    mappings raise ValueError; optional keys use sensible defaults.
    """

    def read_section(self, raw: dict[str, Any]) -> AgglomerativeConfig:
        agg = self.require_mapping(raw, "agglomerative")

        distance_threshold_range: tuple[float, float] | None = None
        if "distance_threshold_range" in agg:
            distance_threshold_range_raw: Any = self.require_value(agg, "distance_threshold_range")
            if not isinstance(distance_threshold_range_raw, (list, tuple)):
                raise ValueError("agglomerative.distance_threshold_range must be a list or tuple with two floats")
            distance_threshold_range_values = cast(Sequence[Any], distance_threshold_range_raw)
            if len(distance_threshold_range_values) != 2:
                raise ValueError("agglomerative.distance_threshold_range must have exactly two values")
            distance_threshold_start = float(distance_threshold_range_values[0])
            distance_threshold_end = float(distance_threshold_range_values[1])
            if distance_threshold_start > distance_threshold_end:
                raise ValueError("agglomerative.distance_threshold_range start must be <= end")
            distance_threshold_range = (distance_threshold_start, distance_threshold_end)
            distance_threshold = distance_threshold_start
        else:
            distance_threshold_raw = agg.get("distance_threshold", None)
            if distance_threshold_raw is not None:
                distance_threshold = float(distance_threshold_raw)
            else:
                distance_threshold = None

        n_clusters: int | None = None
        if "n_clusters" in agg:
            raw_nc = agg["n_clusters"]
            if raw_nc is not None:
                n_clusters = int(raw_nc)

        metric = str(self.optional_value(agg, "metric", "euclidean"))

        linkage = str(self.optional_value(agg, "linkage", "ward"))
        if linkage not in ("ward", "complete", "average", "single"):
            raise ValueError(f"Invalid linkage: {linkage}")

        compute_full_tree = bool(self.optional_value(agg, "compute_full_tree", False))
        n_trials = int(self.require_value(agg, "n_trials"))

        return AgglomerativeConfig(
            distance_threshold=distance_threshold,
            distance_threshold_range=distance_threshold_range,
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage,
            compute_full_tree=compute_full_tree,
            n_trials=n_trials,
        )
