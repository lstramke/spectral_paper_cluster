from __future__ import annotations

from dataclasses import replace

import optuna
import torch

from clustering.affinityPropagation import AffinityPropagationConfig, SklearnAffinityPropagationAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.tfidf import TfidfConfig, TfidfFeatureExtractor
from interpretation.tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

from .pipeline import ExperimentPipeline, MultiRunPipelineResult, PipelineResult, RunSummary


class AffinityPropagationTfidfPipeline(ExperimentPipeline):
    """Pipeline: TF-IDF -> Affinity Propagation -> evaluation/interpretation with Optuna optimization."""

    def __init__(
        self,
        affinityPropagation_config: AffinityPropagationConfig,
        tfidf_config: TfidfConfig,
        interpretation_config: TfidfInterpreterConfig,
    ) -> None:
        self.affinityPropagation_config = affinityPropagation_config
        self.feature_extractor = TfidfFeatureExtractor(tfidf_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        self.interpreter = TfidfInterpreter(interpretation_config)

    def _make_clusterer(self, damping: float, random_state: int):
        # Create adapter with given damping and random_state
        return SklearnAffinityPropagationAdapter(replace(self.affinityPropagation_config, damping=damping, random_state=random_state))

    def _run_single_trial(
        self,
        features,
        damping: float,
        random_state: int,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(damping, random_state)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

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
            metadata={"pipeline": "affinityPropagation_tfidf", "damping": damping, "random_state": random_state},
        )
        return run_summary, pipeline_result

    def run(
        self,
        documents: list[str],
        labels_true: torch.Tensor | None = None,
    ) -> PipelineResult:
        return self.run_many(documents).best_run

    def run_many(
        self,
        documents: list[str],
    ) -> MultiRunPipelineResult:
        """Run AffinityPropagation optimization using Optuna over damping and random_state.
        
        If damping_range in config, optimizes damping over that range.
        If random_state_range in config, optimizes random_state over that range.
        """
        n_trials = self.affinityPropagation_config.n_trials

        if self.affinityPropagation_config.damping_range is not None:
            damping_min, damping_max = self.affinityPropagation_config.damping_range
        else:
            damping_min = damping_max = self.affinityPropagation_config.damping

        if self.affinityPropagation_config.random_state_range is not None:
            random_state_min, random_state_max = self.affinityPropagation_config.random_state_range
        else:
            random_state_min = random_state_max = self.affinityPropagation_config.random_state
            
        features = self.feature_extractor.extract_features(documents)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            damping = trial.suggest_float("damping", damping_min, damping_max)
            random_state = trial.suggest_int("random_state", random_state_min, random_state_max)
            run_summary, pipeline_result = self._run_single_trial(features, damping, random_state)
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
            best_seed=run_summaries[best_idx].seed,
            selected_metric="multi_criteria",
            metadata={"pipeline": "affinityPropagation_tfidf", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
