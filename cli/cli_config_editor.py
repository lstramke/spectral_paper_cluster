from __future__ import annotations

from pathlib import Path
from typing import Any
import io

from ruamel.yaml import YAML
import questionary
from questionary import Style


class CLIConfigEditor:
    """Interactive YAML config editor used by the CLI.

    - Edits are saved immediately after Enter (no backup).
    - Uses `questionary` prompts; accepts an optional `Style` instance.
    - Does not perform schema validation; the project's ConfigReader
      should validate the file when the experiment is executed.
    - Uses ruamel.yaml for round-trip preservation of formatting, comments,
      and flow-style sequences.
    """

    def __init__(self, style: Style | None = None) -> None:
        self.style = style
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False

    def edit_config(self, cfg_path: Path) -> bool:
        """Open the YAML at `cfg_path` and allow interactive edits.

        Returns True if editing completed (saved or exited), False on error.
        """
        if not cfg_path.exists():
            print(f"Config not found: {cfg_path}")
            return False

        with open(cfg_path, encoding="utf-8") as f:
            data = self.yaml.load(f) or {}

        while True:
            sections = list(data.keys()) + ["Exit"]
            sec = questionary.select("Select section:", choices=sections, style=self.style).ask()
            if sec is None or sec == "Exit":
                return True

            value = data.get(sec)
            if isinstance(value, dict):
                fields = list(value.keys()) + ["Back"]
                fld = questionary.select(f"Section `{sec}` - select field:", choices=fields, style=self.style).ask()
                if fld is None or fld == "Back":
                    continue

                cur = value.get(fld)
                new = questionary.text(f"{sec}.{fld}:", default=str(cur) if cur is not None else "", style=self.style).ask()
                parsed = self._yaml_try_parse(new)
                value[fld] = parsed
                data[sec] = value
                with open(cfg_path, "w", encoding="utf-8") as f:
                    self.yaml.dump(data, f)
                print(f"Set {sec}.{fld} = {parsed!r}")
            else:
                cur = value
                new = questionary.text(f"{sec}:", default=str(cur) if cur is not None else "", style=self.style).ask()
                parsed = self._yaml_try_parse(new)
                data[sec] = parsed
                with open(cfg_path, "w", encoding="utf-8") as f:
                    self.yaml.dump(data, f)
                print(f"Set {sec} = {parsed!r}")

    def _yaml_try_parse(self, text: str) -> Any:
        try:
            return self.yaml.load(io.StringIO(text))
        except Exception:
            return text
