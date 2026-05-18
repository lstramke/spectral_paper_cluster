from __future__ import annotations

import re
from typing import List

import questionary
import colorama
from prompt_toolkit.styles import Style

from cli.model.RuleCategory import RuleCategory
from cli.model.cluster import Cluster

from .cluster_summary_repository import ClusterSummaryRepository
from .rule_regex_service import RuleRegexService
from .rule_repository import RuleRepository


colorama.init()


QUESTIONARY_STYLE = Style(
	[
		("separator", "fg:#000000 bold"),
		("question", "fg:#ffffff bold"),
		("instruction", "fg:#b0b0b0"),
		("pointer", "fg:#ffffff"),
		("selected", "fg:#ffffff bold"),
		("answer", "fg:#ffffff bold"),
		("highlighted", "fg:#ffffff bold"),
	]
)


class RuleExtensionController:
	def __init__(
		self,
		rule_repository: RuleRepository,
		summary_repository: ClusterSummaryRepository,
		rule_regex_service: RuleRegexService,
	) -> None:
		self.rule_repository = rule_repository
		self.summary_repository = summary_repository
		self.rule_regex_service = rule_regex_service
		self._compiled_rule_cache: dict[str, re.Pattern | None] = {}
		# staged modifications: category -> RuleCategory (to persist on exit)
		self._modified_rule_categories: dict[str, RuleCategory] = {}

	def run(self):
		aborted = False
		try:
			clusters = self.summary_repository.load_clusters()
			categories = self.rule_repository.list_categories()
			self._print_header("Starting rule review")
			self._print_meta("Summary", str(self.summary_repository.summary_path))
			self._print_meta("Rule root", str(self.rule_repository.rules_root))
			print()

			if not clusters:
				self._print_warning("No clusters found in the summary.")

			if not categories:
				self._print_warning("No top categories found in the rule repository.")

			self._browse_clusters(clusters, categories)

		except KeyboardInterrupt:
			aborted = True
			self._print_warning("Rule review aborted by Ctrl+C.")

		if not aborted and self._modified_rule_categories:
			for cat, rc in self._modified_rule_categories.items():
				try:
					self.rule_repository.save_rules(cat, rc, timestamp=True)
					self._print_success(f"Saved changes for {cat}.")
				except Exception as e:
					self._print_warning(f"Failed to save {cat}: {e}")

	def _browse_clusters(self, clusters: List[Cluster], categories: list[str]) -> None:
		for index, cluster in enumerate(clusters, start=1):
			self._print_cluster(cluster, index, len(clusters))
			selected_categories = questionary.checkbox(
				"Select top categories for this cluster:",
				choices=categories,
				use_arrow_keys=True,
				style=QUESTIONARY_STYLE,
			).ask()
			if selected_categories:
				self._print_success(f"Selected categories: {', '.join(selected_categories)}")
				self._browse_categories(cluster, selected_categories)
			else:
				self._print_warning("No top categories selected for this cluster.")
			print()
			action = questionary.select(
				"Continue to next cluster:",
				choices=["Next", "Exit"],
				use_arrow_keys=True,
				style=QUESTIONARY_STYLE,
			).ask()
			if not action or action == "Exit":
				break

	def _browse_categories(self, cluster: Cluster, selected_categories: list[str]) -> None:
		for category in selected_categories:
			rule_category = self.rule_repository.load_rules(category)
			self._browse_subcategories(cluster, rule_category)

	def _browse_subcategories(self, cluster: Cluster, rule_category: RuleCategory) -> None:
		subcategories = list(rule_category.rules.keys())
		if not subcategories:
			self._print_warning(f"No subcategories found in {rule_category.category}.")
			return

		while True:
			choice = questionary.select(
				f"Select subcategory for {rule_category.category}:",
				choices=subcategories + ["Back"],
				use_arrow_keys=True,
				style=QUESTIONARY_STYLE,
			).ask()
			if not choice or choice == "Back":
				break
			self._review_terms_for_subcategory(cluster, rule_category, choice)

	def _review_terms_for_subcategory(self, cluster: Cluster, rule_category: RuleCategory, subcategory: str) -> None:
		current_rules = rule_category.rules.get(subcategory, [])
		# include any staged rules for this category/subcategory so suggestions
		# don't propose terms already covered by in-memory (staged) additions
		staged = self._modified_rule_categories.get(rule_category.category)
		staged_rules = staged.rules.get(subcategory, []) if staged else []
		combined_rules = current_rules + [r for r in staged_rules if r not in current_rules]
		available_terms = self._available_cluster_terms(cluster, combined_rules)
		if not available_terms:
			self._print_warning("No new cluster terms available for this subcategory.")
			return

		remaining_terms = list(available_terms)
		while remaining_terms:
			choice = questionary.autocomplete(
				"Select a cluster term to add as regex (or Back):",
				choices=remaining_terms + ["Back"],
				style=QUESTIONARY_STYLE,
			).ask()
			if not choice or choice == "Back":
				break

			regex = self.rule_regex_service.suggest_regex(choice)
			self._print_success(f"Suggested regex: {regex}")
			confirm = questionary.confirm(
				"Add this regex to the selected subcategory?",
				default=True,
				style=QUESTIONARY_STYLE,
			).ask()
			if confirm:
				rule_category.add_rule(subcategory, regex)
				self._modified_rule_categories[rule_category.category] = rule_category
				# cache the newly added regex
				try:
					self._compiled_rule_cache[regex] = re.compile(regex)
				except re.error:
					self._print_warning(f"Stored regex is invalid and not cached: {regex}")
				self._print_success(f"Staged '{choice}' for {rule_category.category} / {subcategory}.")
				updated_rules = rule_category.rules.get(subcategory, [])
				remaining_terms = [t for t in remaining_terms if not self._is_term_covered(t, updated_rules) and t != choice]
			else:
				self._print_warning("Skipped this term.")

			remaining_terms = [term for term in remaining_terms if term != choice]

	def _available_cluster_terms(self, cluster: Cluster, existing_rules: list[str]) -> list[str]:
		available_terms: list[str] = []
		for keyword in cluster.keywords:
			for term in self._split_cluster_terms(keyword):
				if self._is_term_covered(term, existing_rules):
					continue
				if term not in available_terms:
					available_terms.append(term)
		return available_terms

	def _is_term_covered(self, term: str, existing_rules: list[str]) -> bool:
		"""Return True if any regex in `existing_rules` matches `term`.

		For readability: for each pattern, ensure a compiled object exists in
		the cache (compile & cache it if missing). Invalid stored patterns
		are warned about and skipped. Then run a single match attempt.
		"""
		for pattern in existing_rules:
			if pattern not in self._compiled_rule_cache:
				try:
					compiled = re.compile(pattern)
				except re.error:
					self._print_warning(f"Invalid stored regex ignored: {pattern}")
					continue
				self._compiled_rule_cache[pattern] = compiled
			else:
				compiled = self._compiled_rule_cache[pattern]

			if not compiled:
				continue

			try:
				if compiled.search(term):
					return True
			except re.error:
				self._print_warning(f"Error running regex: {pattern}")
				continue

		return False

	def _split_cluster_terms(self, keyword: str) -> list[str]:
		parts = [part.strip() for part in re.split(r"[,;]+", keyword) if part and part.strip()]
		return parts or ([keyword.strip()] if keyword.strip() else [])

	def _expand_existing_rule_terms(self, existing_rules: list[str]) -> set[str]:
		existing_terms: set[str] = set()
		for rule in existing_rules:
			for term in re.split(r"[,;]+", rule):
				cleaned_term = term.strip()
				if cleaned_term:
					existing_terms.add(cleaned_term.lower())
		return existing_terms

	def _print_cluster(self, cluster: Cluster, index: int, total: int) -> None:
		print()
		self._print_header(f"[{index}/{total}] Cluster review")
		self._print_meta("Cluster ID", str(cluster.id))
		self._print_keywords("Keywords", ", ".join(cluster.keywords) if cluster.keywords else "-")
		self._print_meta("Documents", str(len(cluster.documents)))

		if not cluster.documents:
			self._print_warning("No documents in this cluster.")

	def _print_header(self, text: str) -> None:
		print(colorama.Style.BRIGHT + colorama.Fore.BLUE + text + colorama.Style.RESET_ALL)

	def _print_meta(self, label: str, value: str) -> None:
		print(colorama.Style.DIM + colorama.Fore.WHITE + f"{label}: " + colorama.Style.BRIGHT + colorama.Fore.WHITE + value + colorama.Style.RESET_ALL)

	def _print_keywords(self, label: str, value: str) -> None:
		print(colorama.Style.BRIGHT + colorama.Fore.CYAN + f"{label}: " + colorama.Style.BRIGHT + colorama.Fore.WHITE + value + colorama.Style.RESET_ALL)

	def _print_warning(self, text: str) -> None:
		print(colorama.Style.DIM + colorama.Fore.YELLOW + text + colorama.Style.RESET_ALL)

	def _print_success(self, text: str) -> None:
		print(colorama.Style.BRIGHT + colorama.Fore.WHITE + text + colorama.Style.RESET_ALL)
