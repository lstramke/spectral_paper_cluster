from typing import cast

from .feature_extractor import FeatureConfig, FeatureExtractor


class FeatureExtractorFactory:
    """Factory that instantiates extractors by name using a config object.

    Supported names (case-insensitive): 'bert', 'fasttext', 'tfidf'. The
    caller must pass a `FeatureConfig` instance of the correct concrete
    subclass (e.g. `BERTConfig`, `FasttextConfig`, `TfidfConfig`). The
    factory will cast and forward the config to the extractor constructor.
    """

    def create(self, name: str, config: FeatureConfig) -> FeatureExtractor:
        """Create an instance of a supported extractor by name.

        Example:
            factory.create('bert', bert_config)
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        key = name.strip().lower()

        if key == "bert":
            from .bert import BertFeatureExtractor, BERTConfig
            if not isinstance(config, BERTConfig):
                raise TypeError("config must be a BERTConfig for extractor 'bert'")
            cfg = cast(BERTConfig, config)
            return BertFeatureExtractor(cfg)

        if key == "fasttext":
            from .fasttext import FasttextFeatureExtractor, FasttextConfig
            if not isinstance(config, FasttextConfig):
                raise TypeError("config must be a FasttextConfig for extractor 'fasttext'")
            cfg = cast(FasttextConfig, config)
            return FasttextFeatureExtractor(cfg)

        if key == "tfidf":
            from .tfidf import TfidfFeatureExtractor, TfidfConfig
            if not isinstance(config, TfidfConfig):
                raise TypeError("config must be a TfidfConfig for extractor 'tfidf'")
            cfg = cast(TfidfConfig, config)
            return TfidfFeatureExtractor(cfg)

        raise ValueError(f"Unknown feature extractor: {name}")
