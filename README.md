# spectral_paper_cluster

Projekt zur Durchführung und Auswertung von Clustering-Experimenten auf Textdaten (TF-IDF basierte Pipelines).

## Projektstruktur

- `config_reader/`
	- Parser für die YAML-Experimentkonfigurationen (Input, Features, Clustering, Output, Interpretation).
- `data/raw/`
	- Rohdaten, z. B. `article_data.csv`, `combined_data.csv`, `web_of_science_search.csv`.
- `data/processed/`
- `experiments/`
	- Pro Experiment ein eigener Ordner, z. B. `kmeans_tfidf/`, `dbscan_tfidf/`, `spectral_tfidf/`.
	- Typischer Inhalt pro Ordner:
		- `<experiment>.py` (ausführbares Skript)
		- `<experiment>.yaml` (Konfiguration)
		- `results_*/` (Run-JSON, Summary, Plot/HTML, spezifische md)
- `src/`
	- `src/clustering/`: Algorithmus-Wrapper (k-means, DBSCAN, OPTICS, HDBSCAN, agglomerative, affinity propagation, spectral).
	- `src/features/`: Feature-Extraktion (TF-IDF).
	- `src/evaluation/`: Unsupervised-Metriken und Evaluator.
	- `src/interpretation/`: Interpretationslogik für Cluster.
	- `src/experiments/`: Gemeinsame Experiment-Hilfsfunktionen und Basisklassen (z. B. `BaseExperiment`, Plot-/IO-Helper).
	- `src/pipelines/`: End-to-End-Pipelines je Experimenttyp.
- `DOCUMENTATION.md`
	- Mathematische Einordnung und Pseudocode der Cluster-Algorithmen und Feature-Extractor.

## Installation

1. Virtuelle Umgebung erstellen und aktivieren.
2. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

## Experimente ausführen

Primär: Interaktives CLI (empfohlen)

Das kleine Hilfsprogramm listet alle Ordner unter `experiments/` auf und startet das gewählte Experiment mit der zugehörigen YAML‑Konfiguration. Dies ist der empfohlene Weg, weil das CLI eine einfache Auswahl, Standardpfade und konsistente Aufrufe bietet.

```bash
python cluster_cli.py
```

Hinweis: Für die Auswahl per Pfeiltasten werden `questionary` und `colorama` verwendet.

Fallback: Einzelnes Experiment per Skript

Wenn du ein Experiment direkt starten möchtest (z. B. für Debugging oder automatisierte Skripte), kannst du das zugehörige Skript mit der YAML-Konfiguration aufrufen:

```bash
python experiments/kmeans_tfidf/kmeans_tfidf.py --config experiments/kmeans_tfidf/kmeans_tfidf.yaml
```

Weitere Beispiele:

- `python experiments/dbscan_tfidf/dbscan_tfidf.py --config experiments/dbscan_tfidf/dbscan_tfidf.yaml`
- `python experiments/hdbscan_tfidf/hdbscan_tfidf.py --config experiments/hdbscan_tfidf/hdbscan_tfidf.yaml`
- `python experiments/spectral_tfidf/spectral_tfidf.py --config experiments/spectral_tfidf/spectral_tfidf.yaml`

Ergebnisse werden im jeweiligen `experiments/<name>/results_*/`-Ordner gespeichert.


## Hinweise

- Die  Dokumentation befindet sich in [`DOCUMENTATION.md`](DOCUMENTATION.md).
- Die folgende Recherche-Query wurde für `web_of_science_search.csv` verwendet:

```text
TS=((hyperspectral imaging OR multispectral imaging)
AND (medical OR clinical OR biomedical OR healthcare))
```