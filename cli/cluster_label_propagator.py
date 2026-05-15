from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from .model.cluster import Cluster
from .model.document import Document


@dataclass
class PropagationResult:
    cluster_id: str | int
    label_name: str
    label_values: Set[str]
    source_doi: str
    updated_documents: int


class ClusterLabelPropagator:
    def __init__(self, clusters: List[Cluster]):
        self.clusters_by_id: Dict[str, Cluster] = {str(cluster.id): cluster for cluster in clusters}
        self.cluster_id_by_doi: Dict[str, str] = {}
        for cluster in clusters:
            cluster_id = str(cluster.id)
            for document in cluster.documents:
                if document.doi:
                    self.cluster_id_by_doi[self._normalize_doi(document.doi)] = cluster_id

    def _normalize_doi(self, doi: str) -> str:
        return doi.strip().lower()

    def get_cluster_for_doi(self, doi: str) -> Optional[Cluster]:
        cluster_id = self.cluster_id_by_doi.get(self._normalize_doi(doi))
        if cluster_id is None:
            return None
        return self.clusters_by_id.get(cluster_id)

    def get_cluster(self, cluster_id: str | int) -> Optional[Cluster]:
        return self.clusters_by_id.get(str(cluster_id))

    def propagate_label(self, cluster_id: str | int, label_name: str, label_values: Iterable[str], source_doi: str) -> PropagationResult:
        cluster = self.clusters_by_id[str(cluster_id)]
        normalized_source_doi = self._normalize_doi(source_doi)
        normalized_values = {str(value).strip() for value in label_values if str(value).strip()}

        updated_documents = 0
        remaining_documents: List[Document] = []
        for document in cluster.documents:
            document_doi = self._normalize_doi(document.doi)
            if document_doi == normalized_source_doi:
                continue

            existing_values = document.labels.setdefault(label_name, set())
            before = len(existing_values)
            existing_values.update(normalized_values)
            if len(existing_values) != before:
                updated_documents += 1
            remaining_documents.append(document)

        cluster.documents = remaining_documents
        return PropagationResult(
            cluster_id=cluster_id,
            label_name=label_name,
            label_values=normalized_values,
            source_doi=source_doi,
            updated_documents=updated_documents,
        )