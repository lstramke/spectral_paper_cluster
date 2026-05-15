from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from questionary import Choice, Separator
import questionary
import colorama
from prompt_toolkit.styles import Style

from cli.model.cluster import Cluster

from .cluster_label_propagator import ClusterLabelPropagator
from .cluster_summary_repository import ClusterSummaryRepository
from .label_repository import LabelCSVReader


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


class PropagationAborted(RuntimeError):
    pass


class LabelPropagationController:
    def __init__(
        self,
        summary_path: Path | str,
        label_csv_path: Path | str,
        output_path: Path | str,
    ) -> None:
        self.summary_repository = ClusterSummaryRepository(summary_path)
        self.label_csv_reader = LabelCSVReader(str(label_csv_path))
        self.output_path = Path(output_path)

    def run(self) -> Path:
        try:
            clusters = self.summary_repository.load_clusters()
            summary_snapshot_path = self.output_path.parent / f"{self.output_path.stem}_cluster_summary.json"
            self.summary_repository.save_summary_json(clusters, summary_snapshot_path)
            batch = self.label_csv_reader.load()
            propagator = ClusterLabelPropagator(clusters)

            source_dois_in_order: list[str] = []

            self._print_header("Starting label propagation")
            self._print_meta("Summary", str(self.summary_repository.summary_path))
            self._print_meta("Snapshot", str(summary_snapshot_path))
            self._print_meta("CSV", str(self.label_csv_reader.path))
            self._print_blank()

            for index, seed_document in enumerate(batch.documents):
                if not seed_document.doi:
                    continue

                cluster = propagator.get_cluster_for_doi(seed_document.doi)
                if cluster is None:
                    self._print_warning(f"[{index}] DOI {seed_document.doi} not found in any cluster, skipping.")
                    continue

                available_label_names = [
                    label_name for label_name, values in seed_document.labels.items() if values
                ]
                if not available_label_names:
                    self._print_warning(f"[{index}] DOI {seed_document.doi} has no usable labels, skipping.")
                    continue

                self._print_blank()
                self._print_cluster_line(index, cluster.id, seed_document.doi)
                self._print_section("Top words")
                if cluster.keywords:
                    self._print_keywords(f"  {', '.join(cluster.keywords)}")
                else:
                    self._print_keywords("  -")
                self._print_blank()
                self._print_meta("Documents in cluster", str(len(cluster.documents)))

                selected_values_by_label = self._choose_label_values_by_area(available_label_names, seed_document)
                if selected_values_by_label is None:
                    raise PropagationAborted()
                if not selected_values_by_label:
                    self._print_warning("Skipping cluster.")
                    continue

                for chosen_label_name in available_label_names:
                    selected_values = selected_values_by_label.get(chosen_label_name)
                    if not selected_values:
                        continue

                    self._print_section(chosen_label_name)
                    self._print_keywords(f"  {', '.join(sorted(selected_values))}")
                    self._print_blank()

                    propagator.propagate_label(
                        cluster_id=cluster.id,
                        label_name=chosen_label_name,
                        label_values=selected_values,
                        source_doi=seed_document.doi,
                    )
                source_dois_in_order.append(seed_document.doi)

            rows = self._build_output_rows(clusters, batch.headers, source_dois_in_order)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.label_csv_reader.save_final_csv(
                out_dir=str(self.output_path.parent),
                rows=rows,
                out_name=self.output_path.name,
                headers=batch.headers,
            )

            self._print_blank()
            self._print_success(f"Wrote propagated labels to {self.output_path}")
            self._print_meta("Rows written", str(len(rows)))
            return self.output_path
        except PropagationAborted:
            self._print_warning("Propagation aborted by user. No further documents will be processed.")
            return self.output_path
        except KeyboardInterrupt:
            self._print_warning("Propagation aborted by Ctrl+C. No further documents will be processed.")
            return self.output_path

    def _build_output_rows(self, clusters: List[Cluster], headers: List[str], source_dois_in_order: Iterable[str]) -> List[dict[str, str]]:
        source_dois = {doi.strip().lower() for doi in source_dois_in_order}
        rows: List[dict[str, str]] = []
        for cluster in clusters:
            for document in cluster.documents:
                if document.doi and document.doi.strip().lower() in source_dois:
                    continue
                if not document.labels:
                    continue
                rows.append(document.to_row(headers))
        return rows

    def _choose_label_values_by_area(self, available_label_names: List[str], seed_document) -> dict[str, set[str]] | None:
        choices: List[Choice | Separator] = []
        for label_name in available_label_names:
            label_values = sorted(seed_document.labels.get(label_name, set()))
            choices.append(Separator(f" {label_name} "))
            for label_value in label_values:
                choices.append(Choice(title=f"  {label_value}", value=(label_name, label_value)))

        selected_values = questionary.checkbox(
            "Select values to propagate:",
            choices=choices,
            use_arrow_keys=True,
            style=QUESTIONARY_STYLE,
        ).ask()
        if selected_values is None:
            return None

        selected_by_label: dict[str, set[str]] = {}
        for label_name, label_value in selected_values:
            selected_by_label.setdefault(label_name, set()).add(label_value)
        return selected_by_label

    def _print_blank(self) -> None:
        print()

    def _print_header(self, text: str) -> None:
        print(colorama.Style.BRIGHT + colorama.Fore.BLUE + text + colorama.Style.RESET_ALL)

    def _print_section(self, text: str) -> None:
        print(colorama.Style.BRIGHT + colorama.Fore.BLUE + text + colorama.Style.RESET_ALL)

    def _print_meta(self, label: str, value: str) -> None:
        print(colorama.Style.DIM + colorama.Fore.WHITE + f"{label}: " + colorama.Style.BRIGHT + colorama.Fore.WHITE + value + colorama.Style.RESET_ALL)

    def _print_keywords(self, text: str) -> None:
        print(colorama.Style.BRIGHT + colorama.Fore.WHITE + text + colorama.Style.RESET_ALL)

    def _print_warning(self, text: str) -> None:
        print(colorama.Style.DIM + colorama.Fore.YELLOW + text + colorama.Style.RESET_ALL)

    def _print_success(self, text: str) -> None:
        print(colorama.Style.BRIGHT + colorama.Fore.WHITE + text + colorama.Style.RESET_ALL)

    def _print_cluster_line(self, index: int, cluster_id: str | int, doi: str) -> None:
        print(
            colorama.Style.BRIGHT
            + colorama.Fore.WHITE
            + f"[{index}] Cluster "
            + colorama.Style.BRIGHT
            + colorama.Fore.WHITE
            + str(cluster_id)
            + colorama.Style.RESET_ALL
            + colorama.Style.DIM
            + colorama.Fore.WHITE
            + " | DOI "
            + colorama.Style.BRIGHT
            + colorama.Fore.WHITE
            + doi
            + colorama.Style.RESET_ALL
        )

