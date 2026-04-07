from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class FeatureExtractionResult:
    """Output contract for text feature extraction."""

    features: torch.Tensor
    feature_names: list[str] = field(default_factory=list[str])
    original_features: torch.Tensor | None = None
    original_feature_names: list[str] = field(default_factory=list[str])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class FeatureExtractor(ABC):
    """Common interface for all text feature extractors."""

    @abstractmethod
    def extract_features(self, documents: list[str]) -> FeatureExtractionResult:
        """Convert raw text documents into a feature matrix."""
