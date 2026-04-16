from __future__ import annotations

import csv
from typing import Protocol, TypeVar, Generic, List
from config_reader.input_config_reader import InputConfig

class HasInput(Protocol):
    input: InputConfig

T = TypeVar("T", bound=HasInput)

class BaseExperiment(Generic[T]):
    
    def load_documents(self, parsed: T) -> List[str]:
        inp = parsed.input
        if inp.format == "line":
            with inp.documents_path.open("r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        docs: List[str] = []
        with inp.documents_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=inp.separator)
            for row in reader:
                values = [str(row.get(field, "")).strip() for field in inp.text_fields]
                non_empty = [v for v in values if v]
                if not non_empty:
                    continue
                if getattr(inp, "fuse_mode", None) == "join":
                    doc = inp.separator.join(non_empty)
                else:
                    doc = non_empty[0]
                docs.append(doc)

        if not docs:
            raise ValueError("No documents found in input file")
        return docs