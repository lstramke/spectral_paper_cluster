from __future__ import annotations

from dataclasses import replace

import optuna
import torch

from clustering.spectralClustering import SpectralClusteringConfig, SklearnSpectralClusteringAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult, RunSummary, MultiRunPipelineResult


class SpectralTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> SPECTRAL -> evaluation/interpretation with Optuna optimization.

    Supports Optuna-based joint optimization of `n_clusters` (int),
    `n_neighbors` (int), and `random_state` (int) when the corresponding `_range` fields are
    present in the `SpectralClusteringConfig`.
    """

    def __init__(
        self,
        spectral_config: SpectralClusteringConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.spectral_config = spectral_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, n_clusters: int, n_neighbors: int, random_state: int) -> SklearnSpectralClusteringAdapter:
        return SklearnSpectralClusteringAdapter(replace(self.spectral_config, n_clusters=n_clusters, n_neighbors=n_neighbors, random_state=random_state))

    def _run_single_trial(
        self,
        features,
        n_clusters: int,
        n_neighbors: int,
        random_state: int,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(n_clusters, n_neighbors, random_state)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

        run_summary = RunSummary(
            seed=random_state,
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
            metadata={"pipeline": "spectral_tfidf", "n_clusters": n_clusters, "n_neighbors": n_neighbors, "random_state": random_state},
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
        """Run SPECTRAL optimization using Optuna over n_clusters, n_neighbors, and random_state.
        
        If n_clusters_range in config, optimizes n_clusters over that range.
        If n_neighbors_range in config, optimizes n_neighbors over that range.
        If random_state_range in config, optimizes random_state over that range.
        """
        n_trials = self.spectral_config.n_trials

        if self.spectral_config.n_clusters_range is not None:
            nc_min, nc_max = self.spectral_config.n_clusters_range
        else:
            nc_min = nc_max = self.spectral_config.n_clusters

        if self.spectral_config.n_neighbors_range is not None:
            nn_min, nn_max = self.spectral_config.n_neighbors_range
        else:
            nn_min = nn_max = self.spectral_config.n_neighbors

        if self.spectral_config.random_state_range is not None:
            rs_min, rs_max = self.spectral_config.random_state_range
        else:
            rs_min = rs_max = self.spectral_config.random_state
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            n_clusters = trial.suggest_int("n_clusters", nc_min, nc_max)
            n_neighbors = trial.suggest_int("n_neighbors", nn_min, nn_max)
            random_state = trial.suggest_int("random_state", rs_min, rs_max)
            run_summary, pipeline_result = self._run_single_trial(features, n_clusters, n_neighbors, random_state)
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
            raise RuntimeError("No SPECTRAL trials were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=run_summaries[best_idx].seed,
            selected_metric="multi_criteria",
            metadata={"pipeline": "spectral_tfidf", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
