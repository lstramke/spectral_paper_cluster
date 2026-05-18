from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RuleCategory:
    category: str
    rules: Dict[str, List[str]] = field(default_factory=dict)

    def add_rule(self, subcategory: str, regex: str) -> None:
        self.rules.setdefault(subcategory, [])
        if regex not in self.rules[subcategory]:
            self.rules[subcategory].append(regex)

    def add_rules(self, subcategory: str, regexes: List[str]) -> None:
        for regex in regexes:
            self.add_rule(subcategory, regex)

    def to_dict(self) -> dict[str, list[str]]:
        return self.rules

    @staticmethod
    def from_dict(category: str, data: dict[str, list[str]]) -> "RuleCategory":
        return RuleCategory(
            category=category,
            rules={name: list(values) for name, values in data.items()}
        )