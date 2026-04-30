from __future__ import annotations


from dataclasses import replace

import optuna

from clustering.kmeans import KMeansConfig, SklearnKMeansAdapter
from evaluation.basic_unsupervised import BasicUnsupervisedEvaluator
from features.bert import BERTConfig, BertFeatureExtractor
from features.feature_extractor import FeatureExtractionResult
from interpretation.bert_interpreter import BertInterpreter, BertInterpreterConfig
from types.document import Document, documents_to_texts

from .pipeline import ExperimentPipeline, PipelineResult, RunSummary, MultiRunPipelineResult


class KmeansBertPipeline(ExperimentPipeline):
    """Pipeline: BERT -> UMAP(100) -> Kmeans -> evaluation/interpretation with Optuna.

    Applies UMAP reduction to 100 dimensions after extracting BERT embeddings.
    Supports Optuna-based joint optimization of `min_cluster_size` and `min_samples`.
    """

    def __init__(
        self,
        kmeans_config: KMeansConfig,
        bert_config: BERTConfig,
        interpretation_config: BertInterpreterConfig,
    ) -> None:
        self.kmeans_config = kmeans_config
        self.feature_extractor = BertFeatureExtractor(bert_config)
        self.evaluator = BasicUnsupervisedEvaluator()
        # Merge model_name from bert_config into the interpretation config if not set
        if interpretation_config.model_name is None:
            interpretation_config = BertInterpreterConfig(
                top_n_terms=interpretation_config.top_n_terms,
                model_name=bert_config.model_name,
                spacy_pipeline=interpretation_config.spacy_pipeline,
                pos_pattern=interpretation_config.pos_pattern,
                use_mmr=interpretation_config.use_mmr,
                diversity=interpretation_config.diversity,
                nr_candidates=interpretation_config.nr_candidates,
            )
        self.interpreter = BertInterpreter(interpretation_config)

    def _make_clusterer(self, seed: int, n_clusters: int):
        # use the sklearn-based KMeans adapter by default
        return SklearnKMeansAdapter(replace(self.kmeans_config, seed=seed, n_clusters=n_clusters))

    def _run_single_seed(
        self,
        features,
        seed: int,
        n_clusters: int,
    ) -> tuple[RunSummary, PipelineResult]:
        clusterer = self._make_clusterer(seed, n_clusters)
        clustering = clusterer.fit_predict(features.features)
        evaluation = self.evaluator.evaluate(features, clustering)

        run_summary = RunSummary(
            seed=seed,
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
            metadata={"pipeline": "kmeans_bert", "seed": seed, "n_clusters": n_clusters},
        )
        return run_summary, pipeline_result

    def run(
        self,
        documents: list[Document],
    ) -> PipelineResult:
        return self.run_many(documents).best_run

    def run_many(
        self,
        documents: list[Document],
    ) -> MultiRunPipelineResult:
        """Run KMeans optimization using Optuna over seed and n_clusters.
        
        If seed_range in config, optimizes seed over that range.
        If n_clusters in config is a tuple (min, max), optimizes over that range.
        """
        n_trials = self.kmeans_config.n_trials

        if self.kmeans_config.seed_range is not None:
            seed_min, seed_max = self.kmeans_config.seed_range
        else:
            seed_min = seed_max = self.kmeans_config.seed

        if self.kmeans_config.cluster_range is not None:
            n_clusters_min, n_clusters_max = self.kmeans_config.cluster_range
        else:
            n_clusters_min = n_clusters_max = self.kmeans_config.n_clusters
            
        texts = documents_to_texts(documents)
        features = self.feature_extractor.extract_features(texts)
        run_summaries: list[RunSummary] = []
        pipeline_results: list[PipelineResult] = []

        def objective(trial: optuna.Trial) -> float:
            seed = trial.suggest_int("seed", seed_min, seed_max)
            n_clusters = trial.suggest_int("n_clusters", n_clusters_min, n_clusters_max)
            run_summary, pipeline_result = self._run_single_seed(features, seed, n_clusters=n_clusters)
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
            document_cluster_mapping = [
                {
                    "doi": document.doi,
                    "text": document.text,
                    "cluster": int(label),
                }
                for document, label in zip(
                    documents,
                    best_result.clustering.labels.detach().cpu().tolist(),
                )
            ]
            best_result.metadata["document_cluster_mapping"] = document_cluster_mapping
        else:
            raise RuntimeError("No seed runs were executed")

        return MultiRunPipelineResult(
            runs=run_summaries,
            best_run=best_result,
            best_seed=run_summaries[best_idx].seed,
            selected_metric="multi_criteria",
            metadata={"pipeline": "kmeans_bert", "n_trials": n_trials, "optuna_seed": 42, "scoring": "silhouette, calinski_harabasz, davies_bouldin"},
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> tuple[float, float, float]:
        silhouette = metrics.get("silhouette", float("-inf"))
        calinski = metrics.get("calinski_harabasz", float("-inf"))
        davies = metrics.get("davies_bouldin", float("inf"))
        return (silhouette, calinski, -davies)
