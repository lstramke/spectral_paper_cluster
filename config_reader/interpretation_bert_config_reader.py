from __future__ import annotations

from typing import Any, Optional

from src.interpretation.bert_interpreter import BertInterpreterConfig
from .config_section_reader import ConfigSectionReader


class InterpretationBertConfigReader(ConfigSectionReader[BertInterpreterConfig]):
    """Reads the `interpretation_bert` section and returns a `BertInterpreterConfig`.

    This keeps BERT-specific interpreter options separate from the
    simple TF-IDF `interpretation` section.
    """

    def read_section(self, raw: dict[str, Any]) -> BertInterpreterConfig:
        interpretation_cfg = self.require_mapping(raw, "interpretation_bert")

        top_n_terms = int(self.require_value(interpretation_cfg, "top_n_terms"))
        if top_n_terms < 1:
            raise ValueError("interpretation_bert.top_n_terms must be >= 1")

        model_name_raw: Optional[Any] = interpretation_cfg.get("model_name")
        model_name: Optional[str] = None
        if model_name_raw is not None:
            model_name = str(model_name_raw)

        spacy_pipeline = str(self.optional_value(interpretation_cfg, "spacy_pipeline", "en_core_web_sm"))
        pos_pattern = str(self.optional_value(interpretation_cfg, "pos_pattern", "<ADJ.*>*<N.*>+"))
        use_mmr = bool(self.optional_value(interpretation_cfg, "use_mmr", False))
        diversity = float(self.optional_value(interpretation_cfg, "diversity", 0.5))

        nr_candidates_raw: Optional[Any] = interpretation_cfg.get("nr_candidates")
        nr_candidates: Optional[int] = None
        if nr_candidates_raw is not None:
            nr_candidates = int(nr_candidates_raw)

        return BertInterpreterConfig(
            top_n_terms=top_n_terms,
            model_name=model_name,
            spacy_pipeline=spacy_pipeline,
            pos_pattern=pos_pattern,
            use_mmr=use_mmr,
            diversity=diversity,
            nr_candidates=nr_candidates,
        )
