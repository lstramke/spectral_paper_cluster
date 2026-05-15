from __future__ import annotations

from dataclasses import dataclass
from typing import Set


@dataclass
class PropagationResult:
    cluster_id: str | int
    label_name: str
    label_values: Set[str]
    source_doi: str
    updated_documents: int
