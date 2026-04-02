from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .feature_extractor import FeatureExtractionResult, FeatureExtractor


@dataclass(slots=True)
class TfidfConfig:
    max_features: int | None
    ngram_range: tuple[int, int]
    min_df: int | float
    max_df: int | float
    lowercase: bool
    use_lsa: bool
    lsa_components: int


class TfidfFeatureExtractor(FeatureExtractor):
    """TF-IDF extractor backed by scikit-learn's TfidfVectorizer."""

    def __init__(self, config: TfidfConfig) -> None:
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=self.config.lowercase,
        )
        self.lsa_pipeline: Any | None = None
        if self.config.use_lsa:
            svd = TruncatedSVD(n_components=self.config.lsa_components)
            normalizer = Normalizer(copy=False)
            self.lsa_pipeline = Pipeline([("svd", svd), ("norm", normalizer)])

    def extract_features(self, documents: list[str]) -> FeatureExtractionResult:
        if not documents:
            raise ValueError("documents must not be empty")

        vectorizer_any = self.vectorizer
        matrix = vectorizer_any.fit_transform(documents)

        if self.lsa_pipeline is not None:
            reduced: Any = self.lsa_pipeline.fit_transform(matrix)
            features = torch.tensor(reduced.tolist(), dtype=torch.float32)
            feature_names = [f"lsa_{i}" for i in range(features.size(1))]
        else:
            features = torch.tensor(matrix.toarray().tolist(), dtype=torch.float32)
            feature_names = list(self.vectorizer.get_feature_names_out())

        return FeatureExtractionResult(
            features=features,
            feature_names=feature_names,
            metadata={
                "extractor": "tfidf",
                "use_lsa": self.config.use_lsa,
                "lsa_components": self.config.lsa_components,
                "n_documents": len(documents),
                "n_features": features.size(1),
            },
        )
