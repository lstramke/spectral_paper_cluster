from __future__ import annotations

from dataclasses import replace

import optuna
import torch

from clustering.gaussianMixture import GMMConfig, SklearnGMMAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, PipelineResult, RunSummary, MultiRunPipelineResult


class GaussianMixtureTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> GaussianMixture -> evaluation/interpretation.

    Supports Optuna-based joint optimization of `n_components` (int)
    and `random_state` (int) when the corresponding `_range` fields are
    present in the `GMMConfig`.
    """

    def __init__(
        self,
        gaussianMixture_config: GMMConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.gaussianMixture_config = gaussianMixture_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, n_components: int, random_state: int) -> SklearnGMMAdapter:
        return SklearnGMMAdapter(replace(self.gaussianMixture_config, n_components=n_components, random_state=random_state))

    def _run_single_trial(
        self,
        features,
        n_components: int,
        random_state: int,
        labels_true: torch.Tensor | None = None,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(n_components=n_components, random_state=random_state)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering, labels_true=labels_true)

        run_summary = RunSummary(
            seed=random_state,
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
            metadata={"pipeline": "gaussianMixture_tfidf", "n_components": n_components, "random_state": random_state},
        )
        return run_summary, pipeline_result

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        return self.run_many(documents, labels_true=labels_true).best_run
    
    def run_many(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> MultiRunPipelineResult:
        """Run GaussianMixture optimization using Optuna over n_components and random_state.
        
        If n_components_range in config, optimizes n_components over that range.
        If random_state_range in config, optimizes random_state over that range.
        """
        n_trials = self.gaussianMixture_config.n_trials

        if self.gaussianMixture_config.n_components_range is not None:
            n_comp_min, n_comp_max = self.gaussianMixture_config.n_components_range
        else:
            n_comp_min = n_comp_max = self.gaussianMixture_config.n_components

        if self.gaussianMixture_config.random_state_range is not None:
            rs_min, rs_max = self.gaussianMixture_config.random_state_range
        else:
            rs_min = rs_max = self.gaussianMixture_config.random_state
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            n_components = trial.suggest_int("n_components", n_comp_min, n_comp_max)
            random_state = trial.suggest_int("random_state", rs_min, rs_max)
            run_summary, pipeline_result = self._run_single_trial(features, n_components=n_components, random_state=random_state, labels_true=labels_true)
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
            best_result.interpretation = self.interpreter.interpret(features, best_result.clustering, labels_true=labels_true)
        else:
            raise RuntimeError("No GMM trials were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=run_summaries[best_idx].seed,
            selected_metric="multi_criteria",
            metadata={"pipeline": "gaussianMixture_tfidf", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )


    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
