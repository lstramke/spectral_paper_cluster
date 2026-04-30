from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    text: str
    doi: str

def documents_to_texts(documents: list[Document]) -> list[str]:
    return [document.text for document in documents]