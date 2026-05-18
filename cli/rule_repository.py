
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

from .model.RuleCategory import RuleCategory


class RuleRepository:
    _RULE_FILE_PATTERN = re.compile(r"^(\d+)rules_(.+)\.json$", re.IGNORECASE)

    def __init__(self, rules_root: str | Path) -> None:
        self.rules_root = Path(rules_root)
        # Read from `input/`, write snapshots to `processed/`.
        self.base_dir = self.rules_root / "input"
        self.processed_dir = self.rules_root / "processed"

    def load_rules(self, category: str) -> RuleCategory:
        rule_file = self._find_rule_file(category)
        if rule_file is None:
            warnings.warn(
                f"No rule file found for '{category}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            return RuleCategory(category=category)

        with rule_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            warnings.warn(
                f"Rule file '{rule_file.name}' does not contain a top-level object.",
                RuntimeWarning,
                stacklevel=2,
            )
            return RuleCategory(category=category)

        normalized_rules: dict[str, list[str]] = {}
        for subcategory, regexes in data.items():
            if isinstance(regexes, list):
                normalized_rules[str(subcategory)] = [str(regex) for regex in regexes]
                continue

            if isinstance(regexes, dict):
                first_list = next((value for value in regexes.values() if isinstance(value, list)), None)
                if first_list is not None:
                    normalized_rules[str(subcategory)] = [str(regex) for regex in first_list]

        return RuleCategory.from_dict(category, normalized_rules)

    def save_rules(self, category: str, data: RuleCategory, timestamp: bool = True) -> Path:
        """Save a snapshot of `data` into the `processed/` directory.

        The method writes only into `processed/` and does not modify files in
        `input/`. A timestamp suffix is added when `timestamp=True`.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        index = self._index_for_category(category)
        base_name = f"{index}rules_{self._normalize_category_name(category)}"
        if timestamp:
            time_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
            filename = f"{base_name}{time_suffix}.json"
        else:
            filename = f"{base_name}.json"

        rule_file = self.processed_dir / filename

        payload = data.to_dict()
        with rule_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        return rule_file

    def list_categories(self) -> list[str]:
        if not self.base_dir.exists():
            return []

        categories: list[tuple[int, str]] = []
        for rule_file in self.base_dir.glob("*.json"):
            match = self._RULE_FILE_PATTERN.match(rule_file.name)
            if match is None:
                continue
            categories.append((int(match.group(1)), match.group(2)))

        categories.sort(key=lambda item: (item[0], item[1]))
        return [category for _, category in categories]

    def _find_rule_file(self, category: str) -> Path | None:
        if not self.base_dir.exists():
            return None

        normalized_category = self._normalize_category_name(category)
        matches: list[tuple[int, Path]] = []
        for rule_file in self.base_dir.glob("*.json"):
            match = self._RULE_FILE_PATTERN.match(rule_file.name)
            if match is None:
                continue
            if self._normalize_category_name(match.group(2)) != normalized_category:
                continue
            matches.append((int(match.group(1)), rule_file))

        if not matches:
            return None

        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    def _index_for_category(self, category: str) -> int:
        """Return the existing numeric index for a category if present in
        `input/` or `processed/`; otherwise return the next available index.
        """
        normalized = self._normalize_category_name(category)
        indices: list[int] = []
        folder = self.base_dir
        if folder.exists():
            for rule_file in folder.glob("*.json"):
                match = self._RULE_FILE_PATTERN.match(rule_file.name)
                if not match:
                    continue
                name = match.group(2)
                if self._normalize_category_name(name) == normalized:
                    try:
                        indices.append(int(match.group(1)))
                    except Exception:
                        continue

        if indices:
            return min(indices)

        return self._next_category_index()

    def _next_category_index(self) -> int:
        max_index = 0
        if self.base_dir.exists():
            for rule_file in self.base_dir.glob("*.json"):
                match = self._RULE_FILE_PATTERN.match(rule_file.name)
                if match is None:
                    continue
                max_index = max(max_index, int(match.group(1)))

        return max_index + 1

    @staticmethod
    def _normalize_category_name(category: str) -> str:
        return category.strip().lower().replace(" ", "_")

