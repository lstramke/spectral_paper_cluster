from __future__ import annotations

from typing import Any, Sequence, cast

from src.features.tfidf import TfidfConfig
from .config_section_reader import ConfigSectionReader


class TfidfConfigReader(ConfigSectionReader[TfidfConfig]):
	"""Reads the `tfidf` section and returns a `TfidfConfig`.

	Expects the raw mapping (the full YAML root) and validates the
	`tfidf` mapping inside it.
	"""

	def read_section(self, raw: dict[str, Any]) -> TfidfConfig:
		tfidf_cfg = self.require_mapping(raw, "tfidf")

		ngram_range_raw: Any = self.require_value(tfidf_cfg, "ngram_range")
		if not isinstance(ngram_range_raw, (list, tuple)):
			raise ValueError("tfidf.ngram_range must be a list with two integers")
		ngram_values = cast(Sequence[Any], ngram_range_raw)
		if len(ngram_values) != 2:
			raise ValueError("tfidf.ngram_range must have exactly two values")

		max_features_raw = tfidf_cfg.get("max_features")
		max_features: int | None = None
		if max_features_raw is not None:
			max_features = int(max_features_raw)

		use_lsa = bool(self.require_value(tfidf_cfg, "use_lsa"))
		lsa_components = 0
		if use_lsa:
			lsa_components = int(self.require_value(tfidf_cfg, "lsa_components"))
			if  lsa_components < 2:
				raise ValueError("tfidf.lsa_components must be >= 2 when LSA is enabled")

		extra_stop_words: list[str] | None = None
		if "extra_stop_words" in tfidf_cfg:
			extra_raw: Any = self.require_value(tfidf_cfg, "extra_stop_words")
			if not isinstance(extra_raw, (list, tuple)):
				raise ValueError("tfidf.extra_stop_words must be a list of strings")
			extra_seq = cast(Sequence[Any], extra_raw)
			extra_stop_words = [str(x) for x in extra_seq]

		return TfidfConfig(
			max_features=max_features,
			ngram_range=(int(ngram_values[0]), int(ngram_values[1])),
			min_df=cast(int | float, self.require_value(tfidf_cfg, "min_df")),
			max_df=cast(int | float, self.require_value(tfidf_cfg, "max_df")),
			lowercase=bool(self.require_value(tfidf_cfg, "lowercase")),
			stop_words=cast(str | list[str] | None, self.require_value(tfidf_cfg, "stop_words")),
			use_lsa=use_lsa,
			lsa_components=lsa_components,
			extra_stop_words=extra_stop_words,
		)
