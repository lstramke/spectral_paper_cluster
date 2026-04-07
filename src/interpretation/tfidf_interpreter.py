from __future__ import annotations

from dataclasses import dataclass

import torch

from clustering.base import ClusteringResult
from features.feature_extractor import FeatureExtractionResult
from .interpreter import ClusterInterpreter
from .interpreter import InterpretationResult


@dataclass(slots=True)
class TfidfInterpreterConfig:
    top_n_terms: int


class TfidfInterpreter(ClusterInterpreter):
    def __init__(self, config: TfidfInterpreterConfig) -> None:
        self.config = config

    def interpret(self, features: FeatureExtractionResult, clustering: ClusteringResult, labels_true: torch.Tensor | None = None,) -> InterpretationResult:
        source_features = features.original_features if features.original_features is not None else features.features
        source_feature_names = (
            features.original_feature_names
            if features.original_feature_names
            else features.feature_names
        )

        labels = clustering.labels.detach().cpu()
        cluster_terms: dict[int, list[tuple[str, float]]] = {}

        for cluster_id in sorted(int(value) for value in labels.unique().tolist()):
            mask = labels == cluster_id
            cluster_matrix = source_features[mask]
            if cluster_matrix.numel() == 0:
                cluster_terms[cluster_id] = []
                continue

            mean_vector = cluster_matrix.mean(dim=0)
            top_indices = torch.topk(mean_vector, k=min(self.config.top_n_terms, mean_vector.numel())).indices
            top_terms = [
                (source_feature_names[int(index)], float(mean_vector[int(index)].item()))
                for index in top_indices.tolist()
            ]

            cluster_terms[cluster_id] = top_terms

        return InterpretationResult(
            cluster_terms=cluster_terms,
            metadata={
                "interpreter": "tfidf",
                "top_n_terms": self.config.top_n_terms,
                "uses_original_features": features.original_features is not None,
                "uses_original_feature_names": bool(features.original_feature_names),
                "labels_true_provided": labels_true is not None,
            },
        )