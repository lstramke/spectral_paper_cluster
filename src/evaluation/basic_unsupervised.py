from __future__ import annotations

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .evaluator import ClusterEvaluator, EvaluationResult
from clustering.base import ClusteringResult
from features.feature_extractor import FeatureExtractionResult


class BasicUnsupervisedEvaluator(ClusterEvaluator):
    """Computes common unsupervised clustering metrics."""

    def evaluate(
        self,
        features: FeatureExtractionResult,
        clustering: ClusteringResult,
        labels_true=None,
    ) -> EvaluationResult:
        x = features.features.detach().cpu().numpy()
        labels = clustering.labels.detach().cpu().numpy()

        unique_labels = set(labels.tolist())
        metrics: dict[str, float] = {}

        # Silhouette and Davies-Bouldin require at least 2 clusters.
        if len(unique_labels) >= 2:
            metrics["silhouette"] = float(silhouette_score(x, labels, metric="cosine"))
            metrics["davies_bouldin"] = float(davies_bouldin_score(x, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, labels))

        return EvaluationResult(
            metrics=metrics,
            metadata={
                "evaluator": "basic_unsupervised",
                "n_clusters_found": clustering.n_clusters_found,
            },
        )
