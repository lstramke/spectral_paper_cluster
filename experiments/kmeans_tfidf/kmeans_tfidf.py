from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import json
import sys
from pathlib import Path
from typing import Any
from time import perf_counter

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import cm, colors as mcolors

# Allow imports from the src package tree when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.interpretation.tfidf_interpreter import TfidfInterpreterConfig
from src.features.tfidf import TfidfConfig
from src.clustering.kmeans import KMeansConfig
from config_reader.input_config_reader import InputConfig
from config_reader.output_config_reader import OutputsConfig
from config_reader.config_reader_new import ConfigReaderBuilder
from src.pipelines.kmeans_tfidf import KMeansTfidfPipeline
from src.pipelines.pipeline import PipelineResult

@dataclass(slots=True)
class ParsedExperimentConfig:
    experiment_name: str
    input: InputConfig
    kmeans: KMeansConfig
    tfidf: TfidfConfig
    interpretation: TfidfInterpreterConfig
    outputs: OutputsConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF + KMeans experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    start_time = perf_counter()
    configReader = ConfigReaderBuilder()\
        .add_input()\
        .add_tfidf()\
        .add_kmeans()\
        .add_interpretation()\
        .add_outputs()\
        .build()
    parsed = configReader.read(config_path)

    # Per-field checks so the type checker (mypy) knows values are non-None.
    if parsed.experiment_name is None:
        print("Missing required config: experiment_name", file=sys.stderr)
        sys.exit(1)
    if parsed.input is None:
        print("Missing required config: input", file=sys.stderr)
        sys.exit(1)
    if parsed.kmeans is None:
        print("Missing required config: kmeans", file=sys.stderr)
        sys.exit(1)
    if parsed.tfidf is None:
        print("Missing required config: tfidf", file=sys.stderr)
        sys.exit(1)
    if parsed.interpretation is None:
        print("Missing required config: interpretation", file=sys.stderr)
        sys.exit(1)
    if parsed.outputs is None:
        print("Missing required config: outputs", file=sys.stderr)
        sys.exit(1)

    experimentConfig = ParsedExperimentConfig(
        experiment_name=parsed.experiment_name or "",
        input=parsed.input,
        kmeans=parsed.kmeans,
        tfidf=parsed.tfidf,
        interpretation=parsed.interpretation,
        outputs=parsed.outputs,
    )

    documents = load_documents(experimentConfig)

    pipeline = KMeansTfidfPipeline(
        kmeans_config=experimentConfig.kmeans,
        tfidf_config=experimentConfig.tfidf,
        interpretation_config=experimentConfig.interpretation,
    )

    multi_run = pipeline.run_many(documents)
    best_result = multi_run.best_run
    plot_path = save_cluster_plot(experimentConfig, best_result)

    try:
        plot_rel = plot_path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        plot_rel = str(plot_path)

    output_dir = Path(experimentConfig.outputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs_path = output_dir / f"{experimentConfig.experiment_name}_all_runs.json"
    best_summary_path = output_dir / experimentConfig.outputs.summary_name

    all_runs_summary: dict[str, Any] = {
        "experiment_name": experimentConfig.experiment_name,
        "seed_range": list(experimentConfig.kmeans.seed_range) if experimentConfig.kmeans.seed_range is not None else [experimentConfig.kmeans.seed],
        "n_seeds": len(multi_run.runs),
        "selected_metric": multi_run.selected_metric,
        "best_seed": multi_run.best_seed,
        "runs": [asdict(run) for run in multi_run.runs],
    }

    best_summary: dict[str, Any] = {
        "experiment_name": experimentConfig.experiment_name,
        "seed": multi_run.best_seed,
        "n_documents": len(documents),
        "n_features": int(best_result.features.features.size(1)),
        "n_clusters_found": best_result.clustering.n_clusters_found,
        "metrics": best_result.evaluation.metrics,
        "objective": best_result.clustering.objective,
        "cluster_sizes": best_result.clustering.cluster_sizes,
        "selected_metric": multi_run.selected_metric,
        "interpretation": asdict(best_result.interpretation) if best_result.interpretation is not None else None,
    }

    elapsed_seconds = perf_counter() - start_time
    all_runs_summary["elapsed_seconds"] = elapsed_seconds
    best_summary["elapsed_seconds"] = elapsed_seconds

    with all_runs_path.open("w", encoding="utf-8") as fp:
        json.dump(all_runs_summary, fp, indent=2)
    with best_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(best_summary, fp, indent=2)

    print(json.dumps(best_summary, indent=2))
    try:
        all_runs_rel = all_runs_path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        all_runs_rel = str(all_runs_path)
    print(f"All runs -> {all_runs_rel}")
    print(f"Plot -> {plot_rel}")
    try:
        summary_rel = best_summary_path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        summary_rel = str(best_summary_path)
    print(f"Summary -> {summary_rel}")

def save_cluster_plot(parsed: ParsedExperimentConfig, result: PipelineResult) -> Path:
    output_dir = parsed.outputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x = result.features.features.detach().cpu().numpy()
    labels = result.clustering.labels.detach().cpu().numpy()

    projection = PCA(n_components=3).fit_transform(x)

    plot_lib = plt
    fig = plot_lib.figure(
        figsize=(parsed.outputs.figsize_width, parsed.outputs.figsize_height)
    )
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        projection[:, 2], # type: ignore[arg-type]
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

    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative as pqual

        x3 = projection[:, 0]
        y3 = projection[:, 1]
        z3 = projection[:, 2]
        html_path = output_path.with_suffix(".html")
        marker_size = max(1, int(parsed.outputs.point_size * 0.15))

        # Create a discrete color for each unique cluster label.
        labels_arr = np.asarray(labels)
        unique_labels = np.unique(labels_arr)
        n_labels = len(unique_labels)

        # Prefer Plotly qualitative palette; fall back to a matplotlib colormap if more colors needed.
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
    return output_path

def load_documents(parsed: ParsedExperimentConfig) -> list[str]:
    if parsed.input.format == "line":
        with parsed.input.documents_path.open("r", encoding="utf-8") as file:
            documents = [line.strip() for line in file if line.strip()]
    else:
        documents: list[str] = []
        with parsed.input.documents_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter=parsed.input.separator)
            for row in reader:
                values = [str(row.get(field, "")).strip() for field in parsed.input.text_fields]
                non_empty = [value for value in values if value]
                if not non_empty:
                    continue

                if parsed.input.fuse_mode == "join":
                    document = parsed.input.separator.join(non_empty)
                else:
                    document = non_empty[0]
                documents.append(document)

    if not documents:
        raise ValueError("No documents found in input file")
    return documents

if __name__ == "__main__":
    main()