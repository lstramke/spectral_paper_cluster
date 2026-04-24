from __future__ import annotations

from dataclasses import replace

import optuna

from clustering.hdbscan import HDBSCANConfig, HDBSCANAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.fasttext import FasttextFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult, RunSummary, MultiRunPipelineResult


class HDBSCANFasttextPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> HDBSCAN -> evaluation/interpretation with Optuna optimization.

    Supports Optuna-based joint optimization of `min_cluster_size` (int)
    and `min_samples` (int) when the corresponding `_range` fields are
    present in the `HDBSCANConfig`.
    """

    def __init__(
        self,
        hdbscan_config: HDBSCANConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.hdbscan_config = hdbscan_config
        self.feature_extractor = FasttextFeatureExtractor()
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, min_cluster_size: int, min_samples: int) -> HDBSCANAdapter:
        return HDBSCANAdapter(replace(self.hdbscan_config, min_cluster_size=min_cluster_size, min_samples=min_samples))

    def _run_single_trial(
        self,
        features,
        min_cluster_size: int,
        min_samples: int,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(min_cluster_size, min_samples)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

        run_summary = RunSummary(
            seed=0,  # Not used for HDBSCAN, but required by RunSummary
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
            metadata={"pipeline": "hdbscan_tfidf", "min_cluster_size": min_cluster_size, "min_samples": min_samples},
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
        """Run HDBSCAN optimization using Optuna over min_cluster_size and min_samples.
        
        If min_cluster_size_range in config, optimizes min_cluster_size over that range.
        If min_samples_range in config, optimizes min_samples over that range.
        Uses multi-criteria scoring for best run selection.
        """
        n_trials = self.hdbscan_config.n_trials

        if self.hdbscan_config.min_cluster_size_range is not None:
            mcs_min, mcs_max = self.hdbscan_config.min_cluster_size_range
        else:
            mcs_min = mcs_max = self.hdbscan_config.min_cluster_size

        if self.hdbscan_config.min_samples_range is not None:
            ms_min, ms_max = self.hdbscan_config.min_samples_range
        else:
            ms_min = ms_max = self.hdbscan_config.min_samples if self.hdbscan_config.min_samples is not None else 1
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            min_cluster_size = trial.suggest_int("min_cluster_size", mcs_min, mcs_max)
            min_samples = trial.suggest_int("min_samples", ms_min, ms_max)
            run_summary, pipeline_result = self._run_single_trial(features, min_cluster_size, min_samples)
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
            raise RuntimeError("No HDBSCAN trials were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=0,  # Not used for HDBSCAN
            selected_metric="multi_criteria",
            metadata={"pipeline": "hdbscan_tfidf", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
