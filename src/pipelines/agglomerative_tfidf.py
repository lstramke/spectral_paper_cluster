from __future__ import annotations

import torch

from clustering.agglomerativeClustering import AgglomerativeConfig, SklearnAgglomerativeAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult


class AgglomerativeTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> Agglomerative Clustering -> evaluation/interpretation."""

    def __init__(
        self,
        agglomerative_config: AgglomerativeConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.agglomerative_config = agglomerative_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        features = self.feature_extractor.extract_features(documents)

        clusterer = SklearnAgglomerativeAdapter(self.agglomerative_config)
        clustering = clusterer.fit_predict(features.features)

        evaluation = self.evaluator.evaluate(features, clustering, labels_true=labels_true)

        interpretation = self.interpreter.interpret(features, clustering, labels_true=labels_true)

        pipeline_result = PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            interpretation=interpretation,
            metadata={"pipeline": "agglomerative_tfidf"},
        )

        return pipeline_result
