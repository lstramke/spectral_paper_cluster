from __future__ import annotations

import torch

from clustering.affinityPropagation import AffinityPropagationConfig, SklearnAffinityPropagationAdapter
from clustering.dbscan import DBSCANConfig, SklearnDBSCANAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult


class AffinityPropagationTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> Affinity Propagation -> evaluation/interpretation."""

    def __init__(
        self,
        affinityPropagation_config: AffinityPropagationConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.affinityPropagation_config = affinityPropagation_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        features = self.feature_extractor.extract_features(documents)

        clusterer = SklearnAffinityPropagationAdapter(self.affinityPropagation_config)
        clustering = clusterer.fit_predict(features.features)

        evaluation = self.evaluator.evaluate(features, clustering, labels_true=labels_true)

        interpretation = self.interpreter.interpret(features, clustering, labels_true=labels_true)

        pipeline_result = PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            interpretation=interpretation,
            metadata={"pipeline": "affinityPropagation_tfidf"},
        )

        return pipeline_result
