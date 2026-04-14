from __future__ import annotations

import torch

from clustering.spectralClustering import SpectralClusteringConfig, SklearnSpectralClusteringAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult


class SpectralTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> SPECTRAL -> evaluation/interpretation."""

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

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        features = self.feature_extractor.extract_features(documents)

        clusterer = SklearnSpectralClusteringAdapter(self.spectral_config)
        clustering = clusterer.fit_predict(features.features)

        evaluation = self.evaluator.evaluate(features, clustering, labels_true=labels_true)

        interpretation = self.interpreter.interpret(features, clustering, labels_true=labels_true)

        pipeline_result = PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            interpretation=interpretation,
            metadata={"pipeline": "spectral_tfidf"},
        )

        return pipeline_result
