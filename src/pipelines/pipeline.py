from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import optuna

from src.clustering.base import ClusteringConfig, ClusteringResult
from src.clustering.clusterer_factory import ClustererFactory
from src.evaluation.evaluator import ClusterEvaluator, EvaluationResult
from src.features.feature_extractor import FeatureExtractionResult, FeatureExtractor
from src.interpretation.interpreter import ClusterInterpreter, InterpretationResult
from src.app_types.document import Document, documents_to_texts


@dataclass(slots=True)
class PipelineResult:
    """End-to-end result of one experiment pipeline run."""

    features: FeatureExtractionResult
    clustering: ClusteringResult
    evaluation: EvaluationResult
    interpretation: InterpretationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(slots=True)
class RunSummary:
    """Compact summary for one seed run."""

    seed: int
    n_clusters_found: int
    metrics: dict[str, float]
    objective: float | None = None
    cluster_sizes: dict[int, int] = field(default_factory=dict[int, int])


@dataclass(slots=True)
class MultiRunPipelineResult:
    """Result of running a pipeline over multiple seeds."""

    runs: list[RunSummary]
    best_run: PipelineResult
    best_seed: int
    selected_metric: str
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

@dataclass(slots=True)
class PipelineConfig:
    """Configuration for a generalized ExperimentPipeline.
    
    Holds all components and settings needed to run a complete pipeline:
    feature extraction, clustering, evaluation, and interpretation.
    """
    
    feature_extractor: FeatureExtractor
    clusterer_factory: ClustererFactory
    clusterer_name: str
    clusterer_config: ClusteringConfig
    evaluator: ClusterEvaluator
    interpreter: ClusterInterpreter | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

class ExperimentPipeline():
    """Concrete, generalized pipeline for all feature/clustering combinations."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with a PipelineConfig."""
        self.config = config

    def run(self, documents: list[Document]) -> PipelineResult:
        """Run pipeline many but return only best run"""
        return self.run_many(documents).best_run

    def run_many(self, documents: list[Document]) -> MultiRunPipelineResult:
        """Run pipeline multiple times for hyperparameter optimization.
        
        Workflow:
        1. Extract features (once)
        2. Optuna optimization loop over clusterer config ranges
        3. Per trial: create clusterer variant → fit_predict → evaluate
        4. Track RunSummary for each trial
        5. Select best run by multi-criteria scoring
        6. Return MultiRunPipelineResult with all runs + best
        """
        texts = documents_to_texts(documents)
        features = self.config.feature_extractor.extract_features(texts)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        n_trials = self.config.clusterer_config.get_n_trials()
        opt_fields = self.config.clusterer_config.get_optimization_fields()
        def objective(trial: optuna.Trial) -> float:
            trial_params = {}
            for field in opt_fields:
                if field.value_type is int:
                        trial_params[field.name] = trial.suggest_int(field.name, field.min_value, field.max_value)
                elif field.value_type is float:
                        trial_params[field.name] = trial.suggest_float(field.name, field.min_value, field.max_value)
            run_summary, pipeline_result = self._run_single_trial(features, trial_params)
            run_summaries.append(run_summary)
            pipeline_results.append(pipeline_result)
            return run_summary.metrics.get("silhouette", -1)
        
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_idx: int | None = None
        best_score: tuple[float, float, float] | None = None

        for idx, run_summary in enumerate(run_summaries):
            current_score = self._score(run_summary.metrics)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_idx = idx

        if best_idx is None:
            raise RuntimeError("No successful trials - cannot select best run")
        
        best_run_summary = run_summaries[best_idx]
        best_pipeline_result = pipeline_results[best_idx]

        if self.config.interpreter is not None:
            best_pipeline_result.interpretation = self.config.interpreter.interpret(
                features, 
                best_pipeline_result.clustering
            )

        document_cluster_mapping = [
            {
                "doi": document.doi,
                "text": document.text,
                "cluster": int(label),
            }
            for document, label in zip(
                documents,
                best_pipeline_result.clustering.labels.detach().cpu().tolist(),
            )
        ]
        best_pipeline_result.metadata["document_cluster_mapping"] = document_cluster_mapping

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_pipeline_result,
            best_seed=best_run_summary.seed,
            selected_metric="multi_criteria",
            metadata={
                "n_trials": n_trials,
                "pipeline": f"{self.config.clusterer_name}_{self.config.feature_extractor.__class__.__name__}",
                "optuna_seed": 42,
                "scoring": "silhouette, calinski_harabasz, davies_bouldin",
            }
        )
        

    def _run_single_trial(
        self,
        features: FeatureExtractionResult,
        trial_params: dict[str, int | float],
    ) -> tuple[RunSummary, PipelineResult]:
        """Execute one trial: create clusterer variant → cluster → evaluate → summarize.""" 
        
        config_variant = replace(self.config.clusterer_config, **trial_params)

        trial_clusterer = self.config.clusterer_factory.create(self.config.clusterer_name, config_variant)

        clustering = trial_clusterer.fit_predict(features.features)
        evaluation = self.config.evaluator.evaluate(features, clustering)

        seed = int(trial_params.get("seed", 0))

        run_summary = RunSummary(
            seed=seed,
            n_clusters_found=clustering.n_clusters_found,
            metrics=dict(evaluation.metrics),
            objective=clustering.objective,
            cluster_sizes=dict(clustering.cluster_sizes)
        )

        pipeline_result = PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            interpretation=None, # only for best run
            metadata={
                "clusterer": self.config.clusterer_name,
                "trial_params": trial_params,
            },
        )

        return run_summary, pipeline_result

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        """Multi-criteria scoring: maximize silhouette & calinski, minimize davies_bouldin."""
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)