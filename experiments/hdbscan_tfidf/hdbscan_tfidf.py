from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import json
import sys
from pathlib import Path
from typing import Any
from time import perf_counter

# Allow imports from the src package tree when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.interpretation.tfidf_interpreter import TfidfInterpreterConfig
from src.features.tfidf import TfidfConfig
from src.clustering.hdbscan import HDBSCANConfig
from config_reader.input_config_reader import InputConfig
from config_reader.output_config_reader import OutputsConfig
from config_reader.config_reader_new import ConfigReaderBuilder
from src.pipelines.hdbscan_tfidf import HDBSCANTfidfPipeline
from src.pipelines.pipeline import PipelineResult
from src.experiments.plot_helper import PlotHelper
from src.experiments.base import BaseExperiment

@dataclass(slots=True)
class ParsedExperimentConfig:
    experiment_name: str
    input: InputConfig
    hdbscan: HDBSCANConfig
    tfidf: TfidfConfig
    interpretation: TfidfInterpreterConfig
    outputs: OutputsConfig


class HDBSCANExperiment(BaseExperiment[ParsedExperimentConfig]):
    """Encapsulates the HDBSCAN + TF-IDF experiment logic."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self.experiment_config: ParsedExperimentConfig | None = None
        self.result = None

    def load_config(self) -> None:
        config_reader = (
            ConfigReaderBuilder()
            .add_input()
            .add_tfidf()
            .add_hdbscan()
            .add_interpretation()
            .add_outputs()
            .build()
        )
        parsed = config_reader.read(self.config_path)

        if parsed.experiment_name is None:
            raise ValueError("Missing required config: experiment_name")
        if parsed.input is None:
            raise ValueError("Missing required config: input")
        if parsed.hdbscan is None:
            raise ValueError("Missing required config: hdbscan")
        if parsed.tfidf is None:
            raise ValueError("Missing required config: tfidf")
        if parsed.interpretation is None:
            raise ValueError("Missing required config: interpretation")
        if parsed.outputs is None:
            raise ValueError("Missing required config: outputs")

        self.experiment_config = ParsedExperimentConfig(
            experiment_name=parsed.experiment_name,
            input=parsed.input,
            hdbscan=parsed.hdbscan,
            tfidf=parsed.tfidf,
            interpretation=parsed.interpretation,
            outputs=parsed.outputs,
        )

    def run(self) -> None:
        if self.experiment_config is None:
            self.load_config()

        assert self.experiment_config is not None

        documents = self.load_documents(self.experiment_config)

        pipeline = HDBSCANTfidfPipeline(
            hdbscan_config=self.experiment_config.hdbscan,
            tfidf_config=self.experiment_config.tfidf,
            interpretation_config=self.experiment_config.interpretation,
        )

        start_time = perf_counter()
        self.result = pipeline.run(documents)
        elapsed_seconds = perf_counter() - start_time

        self._save_results(documents, elapsed_seconds)

    def _save_results(self, documents: list[str], elapsed_seconds: float) -> None:
        assert self.experiment_config is not None
        assert self.result is not None

        PlotHelper.save_cluster_plot(self.experiment_config, self.result)

        output_dir = Path(self.experiment_config.outputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_path = output_dir / f"{self.experiment_config.experiment_name}_run.json"
        summary_path = output_dir / self.experiment_config.outputs.summary_name

        run_summary: dict[str, Any] = {
            "experiment_name": self.experiment_config.experiment_name,
            "n_documents": len(documents),
            "n_features": int(self.result.features.features.size(1)),
            "n_clusters_found": self.result.clustering.n_clusters_found,
            "metrics": self.result.evaluation.metrics,
            "objective": self.result.clustering.objective,
            "cluster_sizes": self.result.clustering.cluster_sizes,
            "interpretation": asdict(self.result.interpretation) if self.result.interpretation is not None else None,
            "elapsed_seconds": elapsed_seconds,
        }

        with run_path.open("w", encoding="utf-8") as fp:
            json.dump(run_summary, fp, indent=2)
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(run_summary, fp, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF + HDBSCAN experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()

    experiment = HDBSCANExperiment(config_path)
    experiment.run()

if __name__ == "__main__":
    main()
