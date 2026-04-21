from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic, List
from pathlib import Path
from time import perf_counter

from config_reader.input_config_reader import InputConfig
from src.pipelines.pipeline import ExperimentPipeline, PipelineResult, MultiRunPipelineResult

class HasInput(Protocol):
    input: InputConfig

T = TypeVar("T", bound=HasInput)

class BaseExperiment(ABC, Generic[T]):
    """Abstract base for experiments.

    Concrete experiments must implement the lifecycle hooks below. A
    reusable `load_documents` implementation is provided.
    """

    config_path: Path
    experiment_config: T | None = None

    def run(self) -> None:
        """Default orchestration for an experiment run using lifecycle hooks.

        Concrete experiments must implement `load_config`, `build_pipeline` and
        `save_results`. This default `run` will call those hooks, execute the
        pipeline and save the result.
        """
        self.load_config()

        assert self.experiment_config is not None

        documents = self.load_documents(self.experiment_config)

        pipeline = self.build_pipeline()

        # execute pipeline and measure elapsed time (inline)
        start = perf_counter()
        result = pipeline.run(documents)
        elapsed = perf_counter() - start

        self.save_results(documents, result, elapsed)

    def run_many(self, seeds: list[int] | None = None) -> None:
        """Orchestrate a multi-run experiment using the pipeline's `run_many`.

        This mirrors `run()` but calls `ExperimentPipeline.run_many` and
        forwards the resulting `MultiRunPipelineResult` to `save_results`.
        """
        self.load_config()

        assert self.experiment_config is not None

        documents = self.load_documents(self.experiment_config)

        pipeline = self.build_pipeline()

        if hasattr(pipeline, "run_many"):
            start = perf_counter()
            result = pipeline.run_many(documents)
            elapsed = perf_counter() - start
            self.save_results(documents, result, elapsed)
        else:
            raise NotImplementedError("Pipeline does not implement run_many()")


    @abstractmethod
    def load_config(self) -> None:
        """Read and validate config; set `self.experiment_config`."""

    @abstractmethod
    def build_pipeline(self) -> ExperimentPipeline:
        """Return an `ExperimentPipeline` instance for this experiment."""

    @abstractmethod
    def save_results(
        self,
        documents: list[str],
        result: PipelineResult | MultiRunPipelineResult,
        elapsed_seconds: float,
    ) -> None:
        """Persist results and any outputs for this experiment.

        Receives the pipeline `result` and measured `elapsed_seconds` so
        concrete experiments don't need to read instance attributes.
        """

    def load_documents(self, parsed: T) -> List[str]:
        inp = parsed.input
        if inp.format == "line":
            with inp.documents_path.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        docs: List[str] = []
        with inp.documents_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=inp.separator)
            for row in reader:
                values = [str(row.get(field, "")).strip() for field in inp.text_fields]
                non_empty = [v for v in values if v]
                if not non_empty:
                    continue
                if getattr(inp, "fuse_mode", None) == "join":
                    doc = inp.separator.join(non_empty)
                else:
                    doc = non_empty[0]
                docs.append(doc)

        if not docs:
            raise ValueError("No documents found in input file")
        return docs
    