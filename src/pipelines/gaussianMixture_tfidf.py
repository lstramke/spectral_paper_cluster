from __future__ import annotations

import torch

from clustering.gaussianMixture import GMMConfig, SklearnGMMAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult


class GaussianMixtureTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> GaussianMixtur -> evaluation/interpretation."""

    def __init__(
        self,
        gaussianMixture_config: GMMConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.gaussianMixture_config = gaussianMixture_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        features = self.feature_extractor.extract_features(documents)

        clusterer = SklearnGMMAdapter(self.gaussianMixture_config)
        clustering = clusterer.fit_predict(features.features)

        evaluation = self.evaluator.evaluate(features, clustering, labels_true=labels_true)

        interpretation = self.interpreter.interpret(features, clustering, labels_true=labels_true)

        pipeline_result = PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            interpretation=interpretation,
            metadata={"pipeline": "gaussianMixture_tfidf"},
        )

        return pipeline_result
