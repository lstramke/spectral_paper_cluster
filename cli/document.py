from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Document:
    title: str = ""
    abstract: str = ""
    doi: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_row(self, headers: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for h in headers:
            if h == "title":
                out[h] = self.title
            elif h == "abstract":
                out[h] = self.abstract
            elif h == "doi":
                out[h] = self.doi
            else:
                out[h] = self.labels.get(h, "")
        return out
    
@dataclass
class DocumentBatch:
    headers: List[str]
    documents: List[Document]