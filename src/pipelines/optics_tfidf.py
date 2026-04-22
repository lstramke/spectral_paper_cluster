from __future__ import annotations

from dataclasses import replace

import optuna

from clustering.optics import OpticsConfig, SklearnOpticsAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult, RunSummary, MultiRunPipelineResult


class OpticsTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> OPTICS -> evaluation/interpretation with Optuna optimization.

    Supports Optuna-based joint optimization of `min_samples` (int)
    and `xi` (float) when the corresponding `_range` fields are
    present in the `OpticsConfig`.
    """

    def __init__(
        self,
        optics_config: OpticsConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.optics_config = optics_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, min_samples: int, xi: float) -> SklearnOpticsAdapter:
        return SklearnOpticsAdapter(replace(self.optics_config, min_samples=min_samples, xi=xi))

    def _run_single_trial(
        self,
        features,
        min_samples: int,
        xi: float,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(min_samples, xi)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

        run_summary = RunSummary(
            seed=0,  # Not used for OPTICS, but required by RunSummary
            n_clusters_found=clustering.n_clusters_found,
            metrics=dict(evaluation.metrics),
            objective=clustering.objective,
            cluster_sizes=dict(clustering.cluster_sizes),
        )

        pipeline_result = PipelineResult(
            features=features,
            clustering=clustering,
            evaluation=evaluation,
            interpretation=None,
            metadata={"pipeline": "optics_tfidf", "min_samples": min_samples, "xi": xi},
        )
        return run_summary, pipeline_result

    def run(
        self,
        documents: list[str],
    ) -> PipelineResult:
        return self.run_many(documents).best_run

    def run_many(
        self,
        documents: list[str],
    ) -> MultiRunPipelineResult:
        """Run OPTICS optimization using Optuna over min_samples and xi.
        
        If min_samples_range in config, optimizes min_samples over that range.
        If xi_range in config, optimizes xi over that range.
        """
        n_trials = self.optics_config.n_trials

        if self.optics_config.min_samples_range is not None:
            ms_min, ms_max = self.optics_config.min_samples_range
        else:
            ms_min = ms_max = self.optics_config.min_samples

        if self.optics_config.xi_range is not None:
            xi_min, xi_max = self.optics_config.xi_range
        else:
            xi_min = xi_max = self.optics_config.xi
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            min_samples = trial.suggest_int("min_samples", ms_min, ms_max)
            xi = trial.suggest_float("xi", xi_min, xi_max)
            run_summary, pipeline_result = self._run_single_trial(features, min_samples, xi)
            run_summaries.append(run_summary)
            pipeline_results.append(pipeline_result)
            
            return run_summary.metrics.get("silhouette", -1)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_idx: int | None = None
        best_score: tuple[float, float, float] | None = None

        for idx, run_summary in enumerate(run_summaries):
            current_score = self._score(run_summary.metrics)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_idx = idx

        if best_idx is not None:
            best_result = pipeline_results[best_idx]
            best_result.interpretation = self.interpreter.interpret(features, best_result.clustering)
        else:
            raise RuntimeError("No OPTICS trials were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=0,  # Not used for OPTICS
            selected_metric="multi_criteria",
            metadata={"pipeline": "optics_tfidf", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
