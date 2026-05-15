from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, TypedDict

from .document import Document


class ExistingLabelEntry(TypedDict):
    doi: str
    labels: Dict[str, Set[str]]


@dataclass
class Cluster:
    """Represents a cluster with its keywords and member documents.

    Fields:
    - `id`: cluster identifier from the summary JSON
    - `keywords`: list of cluster keywords (strings)
    - `documents`: list of `Document` instances belonging to the cluster
    - `existing_labels`: list of mappings indicating which labels came from which DOI
    """
    id: str | int
    keywords: List[str] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    existing_labels: List[ExistingLabelEntry] = field(default_factory=list)

    @staticmethod
    def from_summary_json(
        cluster_id: str | int,
        document_dois: List[str],
        cluster_keywords: List[str],
        document_lookup: Dict[str, Document],
    ) -> Cluster:
        """Factory to build a `Cluster` from summary JSON parts.

        - `cluster_id`: the cluster identifier from the summary JSON
        - `document_dois`: list of DOIs belonging to the cluster
        - `cluster_keywords`: keywords (strings) for the cluster
        - `document_lookup`: mapping DOI -> `Document` (may be partial)

        The factory fills `documents` using `document_lookup`; if a DOI is
        missing from the lookup we create a minimal `Document` with the DOI.
        `existing_labels` is derived from any `Document.labels` present.
        """
        members: List[Document] = []
        for doi in document_dois:
            doc = document_lookup.get(doi)
            if doc is None:
                members.append(Document(title="", abstract="", doi=doi, labels={}))
            else:
                members.append(doc)

        labels_info: List[ExistingLabelEntry] = []
        for doc in members:
            if doc.labels:
                labels_info.append({"doi": doc.doi, "labels": doc.labels})

        return Cluster(id=cluster_id, keywords=cluster_keywords or [], documents=members, existing_labels=labels_info)


