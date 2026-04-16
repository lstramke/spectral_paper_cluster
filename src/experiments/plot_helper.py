from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import cm, colors as mcolors


class PlotHelper:
    """Reusable plotting helper for experiments.

    Usage: from src.experiments.plot_helper import PlotHelper
           PlotHelper.save_cluster_plot(parsed_config, pipeline_result)
    """

    @staticmethod
    def save_cluster_plot(parsed: Any, result: Any) -> None:
        output_dir = Path(parsed.outputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract numpy arrays from tensors
        x = result.features.features.detach().cpu().numpy()
        labels = result.clustering.labels.detach().cpu().numpy()

        # PCA projection to 3D for plotting
        projection = PCA(n_components=3).fit_transform(x)

        fig = plt.figure(figsize=(parsed.outputs.figsize_width, parsed.outputs.figsize_height))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            projection[:, 2],  # type: ignore[arg-type]
            c=labels,
            cmap="tab10",
            s=parsed.outputs.point_size,
            alpha=parsed.outputs.alpha,
        )

        ax.set_title(f"{parsed.experiment_name}")
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.set_zlabel("PCA component 3")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        fig.colorbar(scatter, ax=ax, label="Cluster", shrink=0.75, pad=0.1)
        fig.tight_layout()

        output_path = output_dir / parsed.outputs.plot_name
        fig.savefig(str(output_path), dpi=200)

        # Also write interactive HTML (plotly) if available
        try:
            import plotly.graph_objects as go
            from plotly.colors import qualitative as pqual

            x3 = projection[:, 0]
            y3 = projection[:, 1]
            z3 = projection[:, 2]
            html_path = output_path.with_suffix(".html")
            marker_size = max(1, int(parsed.outputs.point_size * 0.15))

            labels_arr = np.asarray(labels)
            unique_labels = np.unique(labels_arr)
            n_labels = len(unique_labels)

            try:
                base_palette = list(pqual.Plotly)
            except Exception:
                base_palette = []

            if n_labels <= len(base_palette):
                palette = base_palette
            else:
                cmap = cm.get_cmap("tab20", max(n_labels, 20))
                palette = [mcolors.to_hex(cmap(i)) for i in range(n_labels)]

            color_map: dict[int, str] = {}
            for i, lab in enumerate(unique_labels):
                if lab == -1:
                    color_map[int(lab)] = "#AAAAAA"
                else:
                    color_map[int(lab)] = palette[i % len(palette)]

            marker_colors = [color_map[int(l)] for l in labels_arr]

            scatter3d = go.Scatter3d(
                x=x3,
                y=y3,
                z=z3,
                mode="markers",
                marker=dict(size=marker_size, opacity=parsed.outputs.alpha, color=marker_colors),
            )
            figly = go.Figure(data=[scatter3d])
            figly.update_layout(title=parsed.experiment_name)
            figly.write_html(str(html_path), include_plotlyjs="cdn")
        except Exception as e:
            print(f"Could not write interactive HTML plot: {e}", file=sys.stderr)

        plt.close(fig)
