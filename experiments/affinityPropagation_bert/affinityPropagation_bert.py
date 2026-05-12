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

from src.app_types.document import Document
from src.pipelines.pipeline import PipelineResult, MultiRunPipelineResult
from config_reader.config_reader_new import CombinedConfig, ConfigReaderBuilder
from src.experiments.plot_helper import PlotHelper
from src.experiments.base import BaseExperiment

class AffinityPropagationExperiment(BaseExperiment):
    """Encapsulates the AffinityPropagation + bert experiment logic."""
    
    def __init__(self, config_path: str | Path) -> None:
        """Initialize the experiment with a config file path."""
        self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self.experiment_config: CombinedConfig | None = None
    
    def load_config(self) -> None:
        """Load and parse the configuration file."""
        config_reader = (
            ConfigReaderBuilder()
            .add_input()
            .add_bert()
            .add_affinityPropagation()
            .add_interpretation_bert()
            .add_outputs()
            .build()
        )
        self.experiment_config = config_reader.read(self.config_path)
        
        if self.experiment_config.experiment_name is None:
            raise ValueError("Missing required config: experiment_name")
        if self.experiment_config.input is None:
            raise ValueError("Missing required config: input")
        if self.experiment_config.affinityPropagation is None:
            raise ValueError("Missing required config: affinityPropagation")
        if self.experiment_config.bert is None:
            raise ValueError("Missing required config: bert")
        if self.experiment_config.interpretation_bert is None:
            raise ValueError("Missing required config: interpretation_bert")
        if self.experiment_config.outputs is None:
            raise ValueError("Missing required config: outputs")
    
    def save_results(self, documents: list[Document], result: PipelineResult | MultiRunPipelineResult, elapsed_seconds: float) -> None:
        """Save experiment results to output files. Accepts either single or multi-run results."""
        assert self.experiment_config is not None
        assert self.experiment_config.outputs is not None
        assert self.experiment_config.experiment_name is not None

        if isinstance(result, MultiRunPipelineResult):
            pipeline_result = result.best_run
        else:
            pipeline_result = result

        PlotHelper.save_cluster_plot(self.experiment_config, pipeline_result)

        output_dir = Path(self.experiment_config.outputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_path = output_dir / f"{self.experiment_config.experiment_name}_run.json"
        summary_path = output_dir / self.experiment_config.outputs.summary_name

        run_summary: dict[str, Any] = {
            "experiment_name": self.experiment_config.experiment_name,
            "n_documents": len(documents),
            "n_features": int(pipeline_result.features.features.size(1)),
            "n_clusters_found": pipeline_result.clustering.n_clusters_found,
            "metrics": pipeline_result.evaluation.metrics,
            "objective": pipeline_result.clustering.objective,
            "cluster_sizes": pipeline_result.clustering.cluster_sizes,
            "interpretation": asdict(pipeline_result.interpretation) if pipeline_result.interpretation is not None else None,
            "document_cluster_mapping": pipeline_result.metadata.get("document_cluster_mapping") if isinstance(pipeline_result.metadata, dict) else None,
            "elapsed_seconds": elapsed_seconds,
        }

        with run_path.open("w", encoding="utf-8") as fp:
            json.dump(run_summary, fp, indent=2)
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(run_summary, fp, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bert + AffinityPropagation experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    
    experiment = AffinityPropagationExperiment(config_path)
    experiment.run()

if __name__ == "__main__":
    main()
