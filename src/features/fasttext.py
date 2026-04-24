

from __future__ import annotations

from typing import Optional, Any, Dict, Tuple, List, cast

import numpy as np
import torch
import gensim.downloader as api
from gensim.models import KeyedVectors
# using raw fastText embeddings only; no TF-IDF
from dataclasses import dataclass

from .feature_extractor import FeatureExtractionResult, FeatureExtractor

# Default gensim model name (cached by gensim.downloader)
_DEFAULT_MODEL = "fasttext-wiki-news-subwords-300"
_WV: KeyedVectors | None = None

def _get_wv(model_name: str = _DEFAULT_MODEL) -> KeyedVectors:
    """Load and cache KeyedVectors from gensim.downloader.

    On first call this downloads (into gensim cache) and returns the vectors.
    Subsequent calls reuse the cached `KeyedVectors` instance.
    """
    global _WV
    if _WV is not None:
        return _WV
    # api.load will cache the file locally; it returns KeyedVectors
    _WV = cast(KeyedVectors, api.load(model_name))
    return _WV


class FasttextFeatureExtractor(FeatureExtractor):
    """Extract document embeddings by pooling fastText word vectors.

    Parameters
    - model_name: gensim model id (defaults to fasttext wiki-news subwords)
    """

    def __init__(self) -> None:
        self.wv = _get_wv()
        self.dim = self.wv.vector_size

    def extract_features(self, documents: list[str]) -> FeatureExtractionResult:
        if not documents:
            raise ValueError("documents must not be empty")

        rows: list[np.ndarray] = []
        for doc in documents:
            toks: list[str] = [t for t in doc.split() if t in self.wv]
            if not toks:
                rows.append(np.zeros(self.dim, dtype=np.float32))
                continue

            vecs = np.vstack([self.wv[w] for w in toks])
            doc_vec = vecs.mean(axis=0)

            rows.append(doc_vec.astype(np.float32))

        features = torch.from_numpy(np.vstack(rows))
        return FeatureExtractionResult(
            features=features,
            feature_names=[f"ft_{i}" for i in range(features.size(1))],
            metadata={
                "extractor": "fasttext",
                "model_name": _DEFAULT_MODEL,
                "dim": self.dim,
                "use_tfidf": False,
            },
        )