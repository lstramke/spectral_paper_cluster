from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from clustering.base import ClusteringResult
from features.feature_extractor import FeatureExtractionResult


@dataclass(slots=True)
class EvaluationResult:
    """Output contract for clustering evaluation."""

    metrics: dict[str, float] = field(default_factory=dict[str, float])
    artifacts: dict[str, str] = field(default_factory= dict[str, str])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class ClusterEvaluator(ABC):
    """Common interface for clustering evaluators."""

    @abstractmethod
    def evaluate(
        self,
        features: FeatureExtractionResult,
        clustering: ClusteringResult,
        labels_true: torch.Tensor | None = None,
    ) -> EvaluationResult:
        """Evaluate clustering quality.

        labels_true is optional to support unsupervised exploration workflows.
        """
