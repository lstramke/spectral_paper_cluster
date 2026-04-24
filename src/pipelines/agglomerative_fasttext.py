from __future__ import annotations

from dataclasses import replace

import optuna

from clustering.agglomerativeClustering import AgglomerativeConfig, SklearnAgglomerativeAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.fasttext import FasttextFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, MultiRunPipelineResult, PipelineResult, RunSummary


class AgglomerativeFasttextPipeline(ExperimentPipeline):
    """Pipeline: Fasttext -> Agglomerative Clustering -> evaluation/interpretation with Optuna optimization."""

    def __init__(
        self,
        agglomerative_config: AgglomerativeConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.agglomerative_config = agglomerative_config
        self.feature_extractor = FasttextFeatureExtractor()
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, distance_threshold: float):
        # Create adapter with given distance_threshold
        return SklearnAgglomerativeAdapter(replace(self.agglomerative_config, distance_threshold=distance_threshold))

    def _run_single_trial(
        self,
        features,
        distance_threshold: float,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(distance_threshold)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

        run_summary = RunSummary(
            seed=0,  # Not used for agglomerative, but required by RunSummary
            n_clusters_found=clustering.n_clusters_found,
            metrics=dict(evaluation.metrics),
            objective=clustering.objective,
            cluster_sizes=dict(clustering.cluster_sizes),
        )

        pipeline_result = PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            interpretation=None,
            metadata={"pipeline": "agglomerative_tfidf", "distance_threshold": distance_threshold},
        )
        return run_summary, pipeline_result

    def run(
        self,
        documents: list[str],
    ) -> PipelineResult:
        return self.run_many(documents).best_run

    def run_many(
        self,
        documents: list[str],
    ) -> MultiRunPipelineResult:
        """Run Agglomerative optimization using Optuna over distance_threshold.
        
        If distance_threshold_range in config, optimizes over that range.
        """
        n_trials = self.agglomerative_config.n_trials

        # Resolve distance_threshold range
        if self.agglomerative_config.distance_threshold_range is not None:
            threshold_min, threshold_max = self.agglomerative_config.distance_threshold_range
        else:
            # No optimization, run once
            if self.agglomerative_config.distance_threshold is not None:
                threshold_min = threshold_max = self.agglomerative_config.distance_threshold
            else:
                raise ValueError("agglomerative: either distance_threshold or distance_threshold_range must be set")
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            threshold = trial.suggest_float("distance_threshold", threshold_min, threshold_max)
            run_summary, pipeline_result = self._run_single_trial(features, threshold)
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

        if best_idx is not None:
            best_result = pipeline_results[best_idx]
            best_result.interpretation = self.interpreter.interpret(features, best_result.clustering)
        else:
            raise RuntimeError("No trials were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=0,  # Not used for agglomerative
            selected_metric="multi_criteria",
            metadata={"pipeline": "agglomerative_tfidf", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
