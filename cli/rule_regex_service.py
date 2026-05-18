from __future__ import annotations

import re

from cli.model.RegexSuggestion import RegexSuggestion


class RuleRegexService:
    def suggest_regex(self, term: str) -> str:
        cleaned_term = term.strip()
        if not cleaned_term:
            raise ValueError("Term must not be empty")

        parts = [part for part in re.split(r"[\s\-_/]+", cleaned_term) if part]
        if len(parts) <= 1:
            escaped_term = re.escape(cleaned_term)
            return rf"(?i)\b{escaped_term}\b"

        escaped_parts = [re.escape(part) for part in parts]
        joined_parts = r"(?:[\s\-_/]+)".join(escaped_parts)
        return rf"(?i)\b{joined_parts}\b"

    def suggest_regexes(self, terms: list[str]) -> list[RegexSuggestion]:
        suggestions: list[RegexSuggestion] = []
        for term in terms:
            suggestions.append(RegexSuggestion(term=term, regex=self.suggest_regex(term)))
        return suggestions
