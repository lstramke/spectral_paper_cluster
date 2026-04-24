from __future__ import annotations

from typing import cast, Iterable, Optional
from sklearn.decomposition import TruncatedSVD

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
    - min_df: passed to TfidfVectorizer (float for ratio or int for counts)
    - max_df: passed to TfidfVectorizer (float for ratio or int for counts)
    - n_components: reduce pooled embeddings to this many dims using TruncatedSVD
    - extra_stop_words: iterable of additional stop words
    """

    def __init__(
        self,
        min_df: float | int = 0.001,
        max_df: float | int = 0.9,
        n_components: Optional[int] = 100,
        extra_stop_words: Optional[Iterable[str]] = None,
    ) -> None:
        self.wv = _get_wv()
        self.dim = self.wv.vector_size
        self.min_df = min_df
        self.max_df = max_df
        self.n_components = n_components
        self.extra_stop_words = set(extra_stop_words) if extra_stop_words else set()

    def extract_features(self, documents: list[str]) -> FeatureExtractionResult:
        if not documents:
            raise ValueError("documents must not be empty")

        stopwords = set(GENSIM_STOPWORDS) | self.extra_stop_words | {
            "spectral", "hyperspectral", "multispectral", "hsi", "data", "image", "images"
        }

        tf_vectorizer = TfidfVectorizer(stop_words=list(stopwords), min_df=self.min_df, max_df=self.max_df)
        tf_matrix = tf_vectorizer.fit_transform(documents)
        feature_names = np.array(tf_vectorizer.get_feature_names_out())
        allowed_terms = set(feature_names)

        analyzer = tf_vectorizer.build_analyzer()

        rows: list[np.ndarray] = []
        for doc in documents:
            tokens = analyzer(doc)
            vec_list: list[np.ndarray] = []
            for t in tokens:
                if t not in allowed_terms:
                    continue
                vec = None
                for cand in (t, t.lower()):
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

        emb_np = np.vstack(rows)

        if self.n_components is not None and 0 < self.n_components < emb_np.shape[1]:
            svd = TruncatedSVD(n_components=self.n_components, random_state=0)
            reduced = svd.fit_transform(emb_np)
        else:
            reduced = emb_np

        features = torch.from_numpy(np.asarray(reduced, dtype=np.float32))
        features = F.normalize(features, p=2, dim=1)

        original_features = torch.from_numpy(np.asarray(tf_matrix.todense(), dtype=np.float32))

        return FeatureExtractionResult(
            features=features,
            feature_names=[f"ft_{i}" for i in range(features.size(1))],
            original_features=original_features,
            original_feature_names=list(feature_names),
            metadata={
                "extractor": "fasttext",
                "model_name": _DEFAULT_MODEL,
                "orig_dim": self.dim,
                "n_components": self.n_components,
                "use_tfidf": True,
                "tfidf_min_df": self.min_df,
                "tfidf_max_df": self.max_df,
                "tfidf_stop_words": "gensim+extra",
                "extra_stop_words": list(self.extra_stop_words),
                "normalized": True,
            },
        )