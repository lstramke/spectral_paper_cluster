from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any

# Allow imports from the src package tree when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from src.interpretation.interpreter_factory import InterpreterFactory
from src.clustering.clusterer_factory import ClustererFactory
from src.features.feature_extractor_factory import FeatureExtractorFactory
from src.pipelines.pipeline_builder import PipelineBuilder
from src.pipelines.pipeline import PipelineResult, MultiRunPipelineResult, ExperimentPipeline
from config_reader.config_reader_new import CombinedConfig, ConfigReaderBuilder
from src.experiments.plot_helper import PlotHelper
from src.experiments.base import BaseExperiment
from src.app_types.document import Document


class KMeansExperiment(BaseExperiment):
    """Encapsulates the KMeans + TF-IDF multi-run experiment logic."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self.experiment_config: CombinedConfig | None = None
        
    def load_config(self) -> None:
        config_reader = (ConfigReaderBuilder()
            .add_input()
            .add_tfidf()
            .add_kmeans()
            .add_interpretation()
            .add_outputs()
            .build()
        )
        self.experiment_config = config_reader.read(self.config_path)

        if self.experiment_config.experiment_name is None:
            raise ValueError("Missing required config: experiment_name")
        if self.experiment_config.input is None:
            raise ValueError("Missing required config: input")
        if self.experiment_config.kmeans is None:
            raise ValueError("Missing required config: kmeans")
        if self.experiment_config.tfidf is None:
            raise ValueError("Missing required config: tfidf")
        if self.experiment_config.interpretation is None:
            raise ValueError("Missing required config: interpretation")
        if self.experiment_config.outputs is None:
            raise ValueError("Missing required config: outputs")

    def save_results(self, documents: list[Document], result: PipelineResult | MultiRunPipelineResult, elapsed_seconds: float) -> None:
        assert self.experiment_config is not None
        assert self.experiment_config.outputs is not None
        assert self.experiment_config.experiment_name is not None
        assert self.experiment_config.kmeans is not None

        # result is expected to be MultiRunPipelineResult for KMeans
        if isinstance(result, MultiRunPipelineResult):
            multi_run = result
            best_result = multi_run.best_run
        else:
            # fallback: single run
            multi_run = None
            best_result = result

        assert best_result is not None

        PlotHelper.save_cluster_plot(self.experiment_config, best_result)

        output_dir = Path(self.experiment_config.outputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_runs_path = output_dir / f"{self.experiment_config.experiment_name}_all_runs.json"
        best_summary_path = output_dir / self.experiment_config.outputs.summary_name

        all_runs_summary: dict[str, Any] = {
            "experiment_name": self.experiment_config.experiment_name,
            "seed_range": list(self.experiment_config.kmeans.seed_range) if self.experiment_config.kmeans.seed_range is not None else [self.experiment_config.kmeans.seed],
            "n_seeds": len(multi_run.runs) if multi_run is not None else 1,
            "selected_metric": multi_run.selected_metric if multi_run is not None else None,
            "best_seed": multi_run.best_seed if multi_run is not None else None,
            "runs": [asdict(run) for run in multi_run.runs] if multi_run is not None else [],
        }

        best_summary: dict[str, Any] = {
            "experiment_name": self.experiment_config.experiment_name,
            "seed": multi_run.best_seed if multi_run is not None else None,
            "n_documents": len(documents),
            "n_features": int(best_result.features.features.size(1)),
            "n_clusters_found": best_result.clustering.n_clusters_found,
            "metrics": best_result.evaluation.metrics,
            "objective": best_result.clustering.objective,
            "cluster_sizes": best_result.clustering.cluster_sizes,
            "selected_metric": multi_run.selected_metric if multi_run is not None else None,
            "interpretation": asdict(best_result.interpretation) if best_result.interpretation is not None else None,
        }

        all_runs_summary["elapsed_seconds"] = elapsed_seconds
        best_summary["elapsed_seconds"] = elapsed_seconds

        with all_runs_path.open("w", encoding="utf-8") as fp:
            json.dump(all_runs_summary, fp, indent=2)
        with best_summary_path.open("w", encoding="utf-8") as fp:
            json.dump(best_summary, fp, indent=2)


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

    experiment = KMeansExperiment(config_path)
    experiment.run_many()

if __name__ == "__main__":
    main()