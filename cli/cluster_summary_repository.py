
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .model.cluster import Cluster, ExistingLabelEntry
from .model.document import Document


class ClusterSummaryRepository:
	"""Repository for reading clustering summary JSON and producing Cluster objects.

	This repository is instance-based and requires a `summary_path` at
	initialization.
	"""

	def __init__(self, summary_path: Path | str):
		self.summary_path: Path = Path(summary_path)

	def _cluster_sort_key(self, cluster_id: str) -> int:
		return int(cluster_id)


	def load_clusters(self, path: Optional[Path | str] = None) -> List[Cluster]:
		"""Read `path` (or instance path) and return a list of `Cluster` objects.

		The method uses the private `_load_summary()` helper to read the two
		subparts and then constructs `Cluster` objects using the existing
		`Document` dataclass. Labels are left empty.
		"""
		terms, mapping = self._load_summary(path)

		groups: Dict[str, List[Dict[str, Any]]] = {}
		for entry in mapping:
			cluster_id = entry.get("cluster")
			if cluster_id is None:
				continue
			groups.setdefault(str(cluster_id), []).append(entry)

		clusters: List[Cluster] = []
		for cluster_id, cluster_entries in sorted(groups.items(), key=lambda item: self._cluster_sort_key(item[0])):
			keywords = terms.get(cluster_id, [])
			docs: List[Document] = []
			for entry in cluster_entries:
				doi = entry.get("doi") or ""
				text = entry.get("text", "") or ""
				title = text
				abstract = ""
				if ";" in text:
					title_part, rest = text.split(";", 1)
					title = title_part.strip()
					abstract = rest.strip()
				else:
					title = text.strip()
				docs.append(Document(title=title, abstract=abstract, doi=doi, labels={}))

			clusters.append(Cluster(id=cluster_id, keywords=keywords, documents=docs, existing_labels=[]))

		return clusters

	def save_summary_json(self, clusters: List[Cluster], path: Path | str) -> Path:
		summary_payload = [self._cluster_to_payload(cluster) for cluster in clusters]

		output_path = Path(path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8") as fh:
			json.dump(summary_payload, fh, indent=2, ensure_ascii=False)
		return output_path

	def _cluster_to_payload(self, cluster: Cluster) -> Dict[str, Any]:
		return {
			"id": cluster.id,
			"keywords": list(cluster.keywords),
			"dois": [document.doi for document in cluster.documents if document.doi],
		}

	def _load_summary(self, path: Optional[Path | str] = None) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
		"""Private helper: read `path` (or the instance `summary_path`) and
		return `(interpretation_cluster_terms, document_cluster_mapping)`.

		This helper is intentionally private because callers should use the
		higher-level `load_clusters()` API when building `Cluster` objects.
		"""
		p = Path(path) if path is not None else self.summary_path
		if p is None:
			raise ValueError("No summary path provided")
		with p.open("r", encoding="utf-8") as fh:
			summary = json.load(fh)

		interp = summary.get("interpretation", {})
		cluster_terms = interp.get("cluster_terms", {})

		terms_out: Dict[str, List[str]] = {}
		for cluster_id_key, term_entries in cluster_terms.items():
			cluster_id = str(cluster_id_key)
			items = term_entries or []
			term_list: List[str] = []
			for item in items:
				if isinstance(item, list) and len(item) > 0:
					term_list.append(str(item[0]))
				else:
					term_list.append(str(item))
			terms_out[cluster_id] = term_list

		mapping = list(summary.get("document_cluster_mapping", []))
		return terms_out, mapping
