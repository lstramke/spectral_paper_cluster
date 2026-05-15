from __future__ import annotations

from typing import TypedDict


class SummaryDocumentClusterMappingEntry(TypedDict, total=False):
    cluster: str | int
    doi: str
    text: str


class SummaryInterpretation(TypedDict, total=False):
    cluster_terms: dict[str, list[str | list[str]]]


class ClusterSummarySourceJson(TypedDict, total=False):
    interpretation: SummaryInterpretation
    document_cluster_mapping: list[SummaryDocumentClusterMappingEntry]


class ClusterSummaryExportEntry(TypedDict):
    id: str | int
    keywords: list[str]
    dois: list[str]
