from __future__ import annotations

from typing import Any, Sequence, cast, Optional

from src.features.fasttext import FasttextConfig
from .config_section_reader import ConfigSectionReader


class FasttextConfigReader(ConfigSectionReader[FasttextConfig]):
    """Reads the `fasttext` section and returns a `FasttextConfig`.

    Expects the raw mapping (the full YAML root) and validates the
    `fasttext` mapping inside it.
    """

    def read_section(self, raw: dict[str, Any]) -> FasttextConfig:
        fasttext_cfg = self.require_mapping(raw, "fasttext")

        # min_df: can be int or float
        min_df_raw: Any = self.require_value(fasttext_cfg, "min_df")
        try:
            min_df = float(min_df_raw) if isinstance(min_df_raw, float) else int(min_df_raw)
        except (ValueError, TypeError):
            raise ValueError("fasttext.min_df must be a number (int or float)")

        # max_df: can be int or float
        max_df_raw: Any = self.require_value(fasttext_cfg, "max_df")
        try:
            max_df = float(max_df_raw) if isinstance(max_df_raw, float) else int(max_df_raw)
        except (ValueError, TypeError):
            raise ValueError("fasttext.max_df must be a number (int or float)")

        # n_components: optional, defaults to 100
        n_components_raw: Optional[Any] = self.optional_value(fasttext_cfg, "n_components", 100)
        n_components: int | None = None
        if n_components_raw is not None:
            n_components = int(n_components_raw)
            if n_components < 1:
                raise ValueError("fasttext.n_components must be >= 1")

        # extra_stop_words: optional list of strings
        extra_stop_words: list[str] | None = None
        extra_raw: Optional[Any] = self.optional_value(fasttext_cfg, "extra_stop_words")
        if extra_raw is not None:
            if not isinstance(extra_raw, (list, tuple)):
                raise ValueError("fasttext.extra_stop_words must be a list of strings")
            extra_seq = cast(Sequence[Any], extra_raw)
            extra_stop_words = [str(x) for x in extra_seq]

        return FasttextConfig(
            min_df=min_df,
            max_df=max_df,
            n_components=n_components,
            extra_stop_words=extra_stop_words,
        )