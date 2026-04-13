from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_section_reader import ConfigSectionReader


@dataclass(slots=True)
class OutputsConfig:
    output_dir: Path
    plot_name: str
    summary_name: str
    point_size: int
    alpha: float
    figsize_width: float
    figsize_height: float


class OutputsConfigReader(ConfigSectionReader[OutputsConfig]):
    """Reads the `outputs` section and returns an `OutputsConfig`.

    This reader validates the mapping but does not perform filesystem
    side-effects like creating directories; path resolution should be done
    by the caller (e.g. joining with project root) if needed.
    """

    def read_section(self, raw: dict[str, Any]) -> OutputsConfig:
        outputs_cfg = self.require_mapping(raw, "outputs")

        output_dir = Path(str(self.require_value(outputs_cfg, "output_dir")))
        plot_name = str(self.require_value(outputs_cfg, "plot_name"))
        summary_name = str(self.require_value(outputs_cfg, "summary_name"))
        point_size = int(self.require_value(outputs_cfg, "point_size"))
        alpha = float(self.require_value(outputs_cfg, "alpha"))
        figsize_width = float(self.require_value(outputs_cfg, "figsize_width"))
        figsize_height = float(self.require_value(outputs_cfg, "figsize_height"))

        if point_size <= 0:
            raise ValueError("outputs.point_size must be > 0")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("outputs.alpha must be in (0, 1]")
        if figsize_width <= 0 or figsize_height <= 0:
            raise ValueError("outputs.figsize_width and figsize_height must be > 0")

        return OutputsConfig(
            output_dir=output_dir,
            plot_name=plot_name,
            summary_name=summary_name,
            point_size=point_size,
            alpha=alpha,
            figsize_width=figsize_width,
            figsize_height=figsize_height,
        )
