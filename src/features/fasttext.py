

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

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
    _WV = api.load(model_name)
    return _WV


class FasttextFeatureExtractor(FeatureExtractor):
    """Extract document embeddings by pooling fastText word vectors.

    Parameters
    - model_name: gensim model id (defaults to fasttext wiki-news subwords)
    - use_tfidf_weighting: whether to weight token vectors by TF‑IDF
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, use_tfidf_weighting: bool = False) -> None:
        self.model_name = model_name
        self.use_tfidf = use_tfidf_weighting
        self.wv = _get_wv(self.model_name)
        self.dim = self.wv.vector_size

    def extract_features(self, documents: list[str]) -> FeatureExtractionResult:
        if not documents:
            raise ValueError("documents must not be empty")

        tf_vectorizer: Optional[TfidfVectorizer] = None
        tf_matrix = None
        vocab: dict[str, int] = {}
        if self.use_tfidf:
            tf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tf_matrix = tf_vectorizer.fit_transform(documents)
            vocab = {w: idx for idx, w in enumerate(tf_vectorizer.get_feature_names_out())}

        rows: list[np.ndarray] = []
        for doc_idx, doc in enumerate(documents):
            toks = [t for t in doc.split() if t in self.wv]
            if not toks:
                rows.append(np.zeros(self.dim, dtype=np.float32))
                continue

            if self.use_tfidf and tf_matrix is not None:
                weights = []
                for t in toks:
                    idx = vocab.get(t)
                    weight = float(tf_matrix[doc_idx, idx]) if idx is not None else 0.0
                    weights.append(weight)
                weights = np.array(weights, dtype=np.float32)
                vecs = np.vstack([self.wv[w] for w in toks])
                weighted = (vecs * weights[:, None]).sum(axis=0)
                denom = weights.sum() if weights.sum() > 0 else len(toks)
                doc_vec = weighted / denom
            else:
                vecs = np.vstack([self.wv[w] for w in toks])
                doc_vec = vecs.mean(axis=0)

            rows.append(doc_vec.astype(np.float32))

        features = torch.from_numpy(np.vstack(rows))
        return FeatureExtractionResult(
            features=features,
            feature_names=[f"ft_{i}" for i in range(features.size(1))],
            metadata={
                "extractor": "fasttext",
                "model_name": self.model_name,
                "dim": self.dim,
                "use_tfidf": self.use_tfidf,
            },
        )