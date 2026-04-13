from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from .config_section_reader import ConfigSectionReader


@dataclass(slots=True)
class InputConfig:
    documents_path: Path
    format: str
    text_fields: list[str]
    fuse_mode: str
    separator: str


class InputConfigReader(ConfigSectionReader[InputConfig]):
    """Reads the `input` section and returns an `InputConfig`.

    This reader does NOT perform file I/O or path existence checks; it only
    validates and parses the raw mapping. Path resolution (project-root
    joining) should be done by the top-level caller if desired.
    """

    def read_section(self, raw: dict[str, Any]) -> InputConfig:
        input_cfg = self.require_mapping(raw, "input")

        documents_path_raw = self.require_value(input_cfg, "documents_path")
        documents_path = Path(str(documents_path_raw))

        input_format = str(self.require_value(input_cfg, "format"))
        if input_format not in {"line", "csv"}:
            raise ValueError("input.format must be either 'line' or 'csv'")

        text_fields: list[str] = []
        fuse_mode = ""
        separator = ""
        if input_format == "csv":
            text_fields_raw = self.require_value(input_cfg, "text_fields")
            if not isinstance(text_fields_raw, list) or not text_fields_raw:
                raise ValueError("input.text_fields must be a non-empty list for csv format")
            text_fields_seq = cast(Sequence[Any], text_fields_raw)
            text_fields = [str(field) for field in text_fields_seq]

            fuse_mode = str(self.require_value(input_cfg, "fuse_mode"))
            if fuse_mode not in {"join", "first_non_empty"}:
                raise ValueError("input.fuse_mode must be one of: join, first_non_empty")
            separator = str(self.require_value(input_cfg, "separator"))

        return InputConfig(
            documents_path=documents_path,
            format=input_format,
            text_fields=text_fields,
            fuse_mode=fuse_mode,
            separator=separator,
        )
