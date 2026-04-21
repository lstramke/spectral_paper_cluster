from __future__ import annotations

from dataclasses import replace

import optuna
import torch

from clustering.kmeans import KMeansConfig
from clustering.kmeans import SklearnKMeansAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, MultiRunPipelineResult, PipelineResult, RunSummary


class KMeansTfidfPipeline(ExperimentPipeline):
    """First baseline pipeline: TF-IDF -> KMeans -> unsupervised metrics."""

    def __init__(
        self,
        kmeans_config: KMeansConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.kmeans_config = kmeans_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, seed: int, n_clusters: int):
        # use the sklearn-based KMeans adapter by default
        return SklearnKMeansAdapter(replace(self.kmeans_config, seed=seed, n_clusters=n_clusters))

    def _run_single_seed(
        self,
        features,
        seed: int,
        n_clusters: int,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(seed, n_clusters)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

        run_summary = RunSummary(
            seed=seed,
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
            metadata={"pipeline": "kmeans_tfidf", "seed": seed, "n_clusters": n_clusters},
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
        """Run KMeans optimization using Optuna over seed and n_clusters.
        
        If seed_range in config, optimizes seed over that range.
        If n_clusters in config is a tuple (min, max), optimizes over that range.
        """
        n_trials = self.kmeans_config.n_trials

        if self.kmeans_config.seed_range is not None:
            seed_min, seed_max = self.kmeans_config.seed_range
        else:
            seed_min = seed_max = self.kmeans_config.seed

        if self.kmeans_config.cluster_range is not None:
            n_clusters_min, n_clusters_max = self.kmeans_config.cluster_range
        else:
            n_clusters_min = n_clusters_max = self.kmeans_config.n_clusters
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []

        def objective(trial: optuna.Trial) -> float:
            seed = trial.suggest_int("seed", seed_min, seed_max)
            n_clusters = trial.suggest_int("n_clusters", n_clusters_min, n_clusters_max)
            run_summary, _ = self._run_single_seed(features, seed, n_clusters=n_clusters)
            run_summaries.append(run_summary)
            
            return run_summary.metrics.get("silhouette", -1)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_result: PipelineResult | None = None
        best_seed: int | None = None
        best_n_clusters: int | None = None
        best_score: tuple[float, float, float] | None = None

        for run_summary in run_summaries:
            current_score = self._score(run_summary.metrics)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_seed = run_summary.seed

        if study.best_trial:
            best_n_clusters = study.best_trial.params.get("n_clusters", n_clusters_min)
        else:
            best_n_clusters = n_clusters_min

        if best_seed is not None and best_n_clusters is not None:
            _, best_result = self._run_single_seed(features, best_seed, n_clusters=best_n_clusters)
            best_result.interpretation = self.interpreter.interpret(features, best_result.clustering)
            best_result.metadata = {**best_result.metadata, "selected_metric": "silhouette", "best_seed": best_seed, "best_n_clusters": best_n_clusters}
        else:
            raise RuntimeError("No seed runs were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=best_seed,
            selected_metric="silhouette",
            metadata={"pipeline": "kmeans_tfidf", "n_trials": n_trials, "optuna_seed": 42},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)