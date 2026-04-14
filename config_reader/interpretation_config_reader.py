from __future__ import annotations

from typing import Any

from src.interpretation.tfidf_interpreter import TfidfInterpreterConfig
from .config_section_reader import ConfigSectionReader


class InterpretationConfigReader(ConfigSectionReader[TfidfInterpreterConfig]):
	"""Reads the `interpretation` section and returns a `TfidfInterpreterConfig`.

	Expects the raw mapping (the full YAML root) and validates the
	`interpretation` mapping inside it.
	"""

	def read_section(self, raw: dict[str, Any]) -> TfidfInterpreterConfig:
		interpretation_cfg = self.require_mapping(raw, "interpretation")
		top_n_terms = int(self.require_value(interpretation_cfg, "top_n_terms"))
		if top_n_terms < 1:
			raise ValueError("interpretation.top_n_terms must be >= 1")
		return TfidfInterpreterConfig(top_n_terms=top_n_terms)
