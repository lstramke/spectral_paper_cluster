from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from clustering.base import ClusteringResult
from evaluation.evaluator import EvaluationResult
from features.feature_extractor import FeatureExtractionResult


@dataclass(slots=True)
class PipelineResult:
    """End-to-end result of one experiment pipeline run."""

    features: FeatureExtractionResult
    clustering: ClusteringResult
    evaluation: EvaluationResult
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class ExperimentPipeline(ABC):
    """Common interface for experiment pipelines."""

    @abstractmethod
    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        """Run full pipeline: feature extraction -> clustering -> evaluation."""
