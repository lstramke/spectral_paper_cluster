from __future__ import annotations

import torch

from clustering.kmeans import KMeans
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor

from .pipeline import ExperimentPipeline, PipelineResult


class KMeansTfidfPipeline(ExperimentPipeline):
    """First baseline pipeline: TF-IDF -> KMeans -> unsupervised metrics."""

    def __init__(
        self,
        n_clusters: int,
        tfidf_config: TfidfConfig,
        max_iter: int,
        tol: float,
        seed: int,
    ) -> None:
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.clusterer = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        self.evaluator = BasicUnsupervisedEvaluator()

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        features = self.feature_extractor.extract_features(documents)
        clustering = self.clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering, labels_true=labels_true)

        return PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            metadata={"pipeline": "kmeans_tfidf"},
        )
