from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from clustering.base import ClusteringResult
from features.feature_extractor import FeatureExtractionResult


@dataclass(slots=True)
class InterpretationResult:
	"""Output contract for cluster interpretation results."""

	cluster_terms: dict[int, list[tuple[str, float]]] = field(default_factory=dict[int, list[tuple[str, float]]])
	metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class ClusterInterpreter(ABC):
	"""Common interface for all cluster interpreters in this project."""

	@abstractmethod
	def interpret(
		self,
		features: FeatureExtractionResult,
		clustering: ClusteringResult,
		labels_true: torch.Tensor | None = None,
	) -> InterpretationResult:
		"""Interpret clusters and return human-readable descriptors."""
