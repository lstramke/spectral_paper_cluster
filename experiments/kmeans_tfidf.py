from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Allow imports from the src package tree when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from configs.config_reader import ParsedExperimentConfig, load_config, validate_and_parse_config
from pipelines.kmeans_tfidf import KMeansTfidfPipeline


def save_cluster_plot(parsed: ParsedExperimentConfig, result: Any) -> Path:
    output_dir = parsed.outputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x = result.features.features.detach().cpu().numpy()
    labels = result.clustering.labels.detach().cpu().numpy()

    projection = PCA(n_components=2).fit_transform(x)

    plot_lib = plt
    fig, ax = plot_lib.subplots(
        figsize=(parsed.outputs.figsize_width, parsed.outputs.figsize_height)
    )
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        cmap="tab10",
        s=parsed.outputs.point_size,
        alpha=parsed.outputs.alpha,
    )
    ax.set_title(f"{parsed.experiment_name}")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()

    output_path = output_dir / parsed.outputs.plot_name
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
    return output_path


def load_documents(parsed: ParsedExperimentConfig) -> list[str]:
    if parsed.input.format == "line":
        with parsed.input.documents_path.open("r", encoding="utf-8") as file:
            documents = [line.strip() for line in file if line.strip()]
    else:
        documents: list[str] = []
        with parsed.input.documents_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
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


def build_pipeline(parsed: ParsedExperimentConfig) -> KMeansTfidfPipeline:
    return KMeansTfidfPipeline(
        n_clusters=parsed.pipeline_n_clusters,
        tfidf_config=parsed.tfidf,
        max_iter=parsed.pipeline_max_iter,
        tol=parsed.pipeline_tol,
        seed=parsed.pipeline_seed,
    )


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
    config = load_config(config_path)
    parsed = validate_and_parse_config(config, PROJECT_ROOT)
    documents = load_documents(parsed)
    pipeline = build_pipeline(parsed)

    result = pipeline.run(documents)
    plot_path = save_cluster_plot(parsed, result)

    try:
        plot_rel = plot_path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        plot_rel = str(plot_path)

    output_dir = Path(parsed.outputs.output_dir)
    summary_path = output_dir / parsed.outputs.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        summary_rel = summary_path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        summary_rel = str(summary_path)

    summary: dict[str, Any] = {
        "experiment_name": parsed.experiment_name,
        "n_documents": len(documents),
        "n_features": int(result.features.features.size(1)),
        "n_clusters_found": result.clustering.n_clusters_found,
        "metrics": result.evaluation.metrics,
    }

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Plot -> {plot_rel}")
    print(f"Summary -> {summary_rel}")


if __name__ == "__main__":
    main()
