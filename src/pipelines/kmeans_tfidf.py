from __future__ import annotations

from dataclasses import replace

import torch

from clustering.kmeans import KMeans, KMeansConfig
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
        interpretation_config: TfidfInterpreterConfig | None = None,
    ) -> None:
        self.kmeans_config = kmeans_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, seed: int) -> KMeans:
        return KMeans(replace(self.kmeans_config, seed=seed))

    def _run_single_seed(
        self,
        features,
        seed: int,
        labels_true: torch.Tensor | None = None,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(seed)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering, labels_true=labels_true)

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
            metadata={"pipeline": "kmeans_tfidf", "seed": seed},
        )
        return run_summary, pipeline_result

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        return self.run_many(documents, labels_true=labels_true).best_run

    def run_many(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
        seeds: list[int] | None = None,
    ) -> MultiRunPipelineResult:
        if seeds is None or not seeds:
            if self.kmeans_config.seed_range is not None:
                start, end = self.kmeans_config.seed_range
                seeds = list(range(start, end + 1))
            else:
                seeds = [self.kmeans_config.seed]

        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        best_result: PipelineResult | None = None
        best_seed: int | None = None
        best_score: tuple[float, float, float] | None = None

        for seed in seeds:
            run_summary, pipeline_result = self._run_single_seed(features, seed, labels_true=labels_true)
            run_summaries.append(run_summary)

            current_score = self._score(run_summary.metrics)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_result = pipeline_result
                best_seed = seed

        if best_result is None or best_seed is None:
            raise RuntimeError("No seed runs were executed")

        best_result.interpretation = self.interpreter.interpret(features, best_result.clustering, labels_true=labels_true)
        best_result.metadata = {**best_result.metadata, "selected_metric": "silhouette", "best_seed": best_seed}

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=best_seed,
            selected_metric="silhouette",
            metadata={"pipeline": "kmeans_tfidf", "n_seeds": len(seeds)},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)