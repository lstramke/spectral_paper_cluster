from __future__ import annotations

from typing import Any, Optional

from src.features.bert import BERTConfig
from .config_section_reader import ConfigSectionReader


class BertConfigReader(ConfigSectionReader[BERTConfig]):
    """Reads the `bert` section and returns a `BERTConfig`.

    Expects a mapping under the top-level `bert` key with the necessary
    fields. Validates types and some basic ranges.
    """

    def read_section(self, raw: dict[str, Any]) -> BERTConfig:
        bert_cfg = self.require_mapping(raw, "bert")

        model_name = str(self.require_value(bert_cfg, "model_name"))
        device = str(self.require_value(bert_cfg, "device"))
        batch_size = int(self.require_value(bert_cfg, "batch_size"))
        if batch_size < 1:
            raise ValueError("bert.batch_size must be >= 1")

        normalize = bool(self.require_value(bert_cfg, "normalize"))
        show_progress = bool(self.require_value(bert_cfg, "show_progress"))

        # umap_n_components may be omitted or set to null to disable
        umap_raw: Optional[Any] = bert_cfg.get("umap_n_components")
        umap_n_components: int | None
        if umap_raw is None:
            umap_n_components = None
        else:
            umap_n_components = int(umap_raw)

        umap_random_state = int(self.require_value(bert_cfg, "umap_random_state"))

        preprocess_with_tfidf = bool(bert_cfg.get("preprocess_with_tfidf", False))
        tfidf_max_df_raw = bert_cfg.get("tfidf_max_df", 0.5)
        try:
            tfidf_max_df = float(tfidf_max_df_raw)
        except Exception:
            raise ValueError("bert.tfidf_max_df must be a float")

        tfidf_max_features_raw = bert_cfg.get("tfidf_max_features", None)
        tfidf_max_features: int
        if tfidf_max_features_raw is None:
            tfidf_max_features = 0
        else:
            tfidf_max_features = int(tfidf_max_features_raw)

        return BERTConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            show_progress=show_progress,
            umap_n_components=umap_n_components,
            umap_random_state=umap_random_state,
            preprocess_with_tfidf=preprocess_with_tfidf,
            tfidf_max_df=tfidf_max_df,
            tfidf_max_features=tfidf_max_features,
        )
