from __future__ import annotations

from dataclasses import replace

import optuna

from clustering.dbscan import DBSCANConfig, SklearnDBSCANAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.fasttext import FasttextFeatureExtractor
from features.feature_extractor import FeatureExtractionResult
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig


from .pipeline import ExperimentPipeline, MultiRunPipelineResult, PipelineResult, RunSummary


class DBSCANFasttextPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> DBSCAN -> evaluation/interpretation with Optuna optimization."""

    def __init__(
        self,
        dbscan_config: DBSCANConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.dbscan_config = dbscan_config
        self.feature_extractor = FasttextFeatureExtractor()
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, eps: float, min_samples: int):
        # Create adapter with given eps and min_samples
        return SklearnDBSCANAdapter(replace(self.dbscan_config, eps=eps, min_samples=min_samples))

    def _run_single_trial(
        self,
        features: FeatureExtractionResult,
        eps: float,
        min_samples: int,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(eps, min_samples)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

        run_summary = RunSummary(
            seed=0,  # Not used for DBSCAN, but required by RunSummary
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
            metadata={"pipeline": "dbscan_tfidf", "eps": eps, "min_samples": min_samples},
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
        """Run DBSCAN optimization using Optuna over eps and min_samples.
        
        If eps_range in config, optimizes eps over that range.
        If min_samples_range in config, optimizes min_samples over that range.
        """
        n_trials = self.dbscan_config.n_trials

        if self.dbscan_config.eps_range is not None:
            eps_min, eps_max = self.dbscan_config.eps_range
        else:
            eps_min = eps_max = self.dbscan_config.eps

        if self.dbscan_config.min_samples_range is not None:
            min_samples_min, min_samples_max = self.dbscan_config.min_samples_range
        else:
            min_samples_min = min_samples_max = self.dbscan_config.min_samples
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            eps = trial.suggest_float("eps", eps_min, eps_max)
            min_samples = trial.suggest_int("min_samples", min_samples_min, min_samples_max)
            run_summary, pipeline_result = self._run_single_trial(features, eps, min_samples)
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
            raise RuntimeError("No trials were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=0,  # Not used for DBSCAN
            selected_metric="multi_criteria",
            metadata={"pipeline": "dbscan_tfidf", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
