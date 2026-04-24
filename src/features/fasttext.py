

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F

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

        tf_vectorizer = TfidfVectorizer(stop_words=list(GENSIM_STOPWORDS))
        tf_matrix = tf_vectorizer.fit_transform(documents)
        feature_names = list(tf_vectorizer.get_feature_names_out())
        
        for doc in documents:
            tokens = doc.split()
            vec_list: list[np.ndarray] = []
            for t in tokens:
                t_norm = t.lower()
                if t_norm in GENSIM_STOPWORDS:
                    continue
                vec = None
                for cand in (t, t_norm):
                    try:
                        vec = self.wv[cand]
                        break
                    except Exception:
                        try:
                            vec = self.wv.get_vector(cand)  # type: ignore[attr-defined]
                            break
                        except Exception:
                            continue

                if vec is None:
                    continue

                vec_list.append(np.asarray(vec, dtype=np.float32))

            if not vec_list:
                rows.append(np.zeros(self.dim, dtype=np.float32))
                continue

            vecs = np.vstack(vec_list)
            doc_vec = vecs.mean(axis=0)

            rows.append(doc_vec.astype(np.float32))

        features = torch.from_numpy(np.vstack(rows))
        features = F.normalize(features, p=2, dim=1)

        original_features = torch.from_numpy(np.asarray(tf_matrix, dtype=np.float32))

        return FeatureExtractionResult(
            features=features,
            feature_names=[f"ft_{i}" for i in range(features.size(1))],
            original_features=original_features,
            original_feature_names=feature_names,
            metadata={
                "extractor": "fasttext",
                "model_name": _DEFAULT_MODEL,
                "dim": self.dim,
                "use_tfidf": False,
                "normalized": True,
                "tfidf_stop_words": "gensim",
            },
        )