from __future__ import annotations

from typing import Any

from src.clustering.agglomerativeClustering import AgglomerativeConfig
from .config_section_reader import ConfigSectionReader


class AgglomerativeConfigReader(ConfigSectionReader[AgglomerativeConfig]):
    """Reads the `agglomerative` section and returns an `AgglomerativeConfig`.

    Follows the same style as `DbscanConfigReader`: missing or invalid
    mappings raise ValueError; optional keys use sensible defaults.
    """

    def read_section(self, raw: dict[str, Any]) -> AgglomerativeConfig:
        agg = self.require_mapping(raw, "agglomerative")

        # n_clusters may be null/None for distance_threshold-based cuts
        n_clusters: int | None = None
        if "n_clusters" in agg:
            raw_nc = agg["n_clusters"]
            if raw_nc is not None:
                n_clusters = int(raw_nc)

        metric = str(self.optional_value(agg, "metric", "euclidean"))

        linkage = str(self.optional_value(agg, "linkage", "ward"))
        if linkage not in ("ward", "complete", "average", "single"):
            raise ValueError(f"Invalid linkage: {linkage}")

        distance_threshold = agg.get("distance_threshold", None)
        if distance_threshold is not None:
            distance_threshold = float(distance_threshold)

        compute_full_tree = bool(self.optional_value(agg, "compute_full_tree", False))

        return AgglomerativeConfig(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_full_tree=compute_full_tree,
        )
