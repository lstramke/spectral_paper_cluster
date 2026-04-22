from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from clustering.base import ClusteringResult
from evaluation.evaluator import EvaluationResult
from features.feature_extractor import FeatureExtractionResult
from interpretation.interpreter import InterpretationResult


@dataclass(slots=True)
class PipelineResult:
    """End-to-end result of one experiment pipeline run."""

    features: FeatureExtractionResult
    clustering: ClusteringResult
    evaluation: EvaluationResult
    interpretation: InterpretationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(slots=True)
class RunSummary:
    """Compact summary for one seed run."""

    seed: int
    n_clusters_found: int
    metrics: dict[str, float]
    objective: float | None = None
    cluster_sizes: dict[int, int] = field(default_factory=dict[int, int])


@dataclass(slots=True)
class MultiRunPipelineResult:
    """Result of running a pipeline over multiple seeds."""

    runs: list[RunSummary]
    best_run: PipelineResult
    best_seed: int
    selected_metric: str
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class ExperimentPipeline(ABC):
    """Common interface for experiment pipelines."""

    @abstractmethod
    def run(
        self,
        documents: list[str]
    ) -> PipelineResult:
        """Run full pipeline: feature extraction -> clustering -> evaluation."""

    @abstractmethod
    def run_many(
        self,
        documents: list[str]
    ) -> MultiRunPipelineResult:
        """Run the pipeline multiple times for a seed list."""
        raise NotImplementedError("run_many() is not implemented for this pipeline")
