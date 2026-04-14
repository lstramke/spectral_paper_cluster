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
		- `<experiment>.md` (Beschreibung)
		- `outputs/` (Run-JSON, Summary, Plot/HTML)
- `src/`
	- `src/clustering/`: Algorithmus-Wrapper (k-means, DBSCAN, OPTICS, HDBSCAN, agglomerative, affinity propagation, spectral).
	- `src/features/`: Feature-Extraktion (TF-IDF).
	- `src/evaluation/`: Unsupervised-Metriken und Evaluator.
	- `src/interpretation/`: Interpretationslogik für Cluster.
	- `src/pipelines/`: End-to-End-Pipelines je Experimenttyp.
- `EXPERIMENTS.md`
	- Mathematische Einordnung, Pseudocode und inhaltliche Beschreibung der Experimente.

## Installation

1. Virtuelle Umgebung erstellen und aktivieren.
2. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

## Experimente ausführen

Die Skripte werden jeweils mit der zugehörigen YAML-Konfiguration gestartet.

Beispiel:

```bash
python experiments/kmeans_tfidf/kmeans_tfidf.py --config experiments/kmeans_tfidf/kmeans_tfidf.yaml
```

Weitere Beispiele:

- `python experiments/dbscan_tfidf/dbscan_tfidf.py --config experiments/dbscan_tfidf/dbscan_tfidf.yaml`
- `python experiments/hdbscan_tfidf/hdbscan_tfidf.py --config experiments/hdbscan_tfidf/hdbscan_tfidf.yaml`
- `python experiments/spectral_tfidf/spectral_tfidf.py --config experiments/spectral_tfidf/spectral_tfidf.yaml`

Ergebnisse werden im jeweiligen `experiments/<name>/outputs/`-Ordner gespeichert.

## Hinweise

- Die  Dokumentation befindet sich in `DOCUMENTATION.md`.
- Die folgende Recherche-Query wurde für `web_of_science_search.csv` verwendet:

```text
TS=((hyperspectral imaging OR multispectral imaging)
AND (medical OR clinical OR biomedical OR healthcare))
```