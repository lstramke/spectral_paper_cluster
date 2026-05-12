# Gesamtstruktur des Projekts

Das Projekt ist so aufgebaut, dass Experimente möglichst direkt aus der Ordnerstruktur heraus nachvollziehbar und erweiterbar sind. Die zentrale Trennung liegt zwischen Konfiguration, Daten, Experiment-Definitionen und der eigentlichen Implementierungslogik in `src/`.

## 1. Komponenten eines Experiments

Ein Experiment besteht nicht nur aus dem Skript im jeweiligen Ordner, sondern aus mehreren klar getrennten Bausteinen. Im Zentrum steht dabei `BaseExperiment` aus [src/experiments/base.py](src/experiments/base.py): Die Klasse stellt mit `run()` einen allgemeinen Ablauf bereit und kapselt die gemeinsamen Schritte für alle Experimente.

Die konkrete Ausführung nutzt zentrale Hooks und eine gemeinsame Pipeline-Instanziierung:

- `load_config()`
  - abstrakter Hook: baut den passenden `ConfigReader` (Builder-Pattern), lädt und validiert die YAML- Konfiguration und setzt die experiment-spezifische Config-Dataclass

- `build_pipeline()`
  - ist in `BaseExperiment` implementiert: die Methode verwendet den `PipelineBuilder` um aus dem geladenen `CombinedConfig` ein `ExperimentPipeline` zu erzeugen. Sie kapselt die Erkennung von Feature-Extractor, Clusterer und Interpreter und übergibt die Komponenten an die standardisierte Pipeline. Siehe [src/experiments/base.py](src/experiments/base.py#L1-L200), [src/pipelines/pipeline_builder.py](src/pipelines/pipeline_builder.py#L1-L200) und [src/pipelines/pipeline.py](src/pipelines/pipeline.py#L1-L200).

- `save_results()`
  - abstrakter Hook: bestimmt, wie Ausgaben, JSON-Dateien, Plots oder Summarys gespeichert werden und kann experimentabhängig angepasst werden

`BaseExperiment.run()` nutzt diese Schritte dann für den allgemeinen Ablauf: Konfiguration laden, Dokumente einlesen, Pipeline bauen (via `PipelineBuilder`), Pipeline ausführen und Ergebnisse speichern. Dadurch bleibt die gemeinsame Logik zentral, während einzelne Experimente nur ihre eigenen Details (Konfig-Lesen und Ergebnispersistenz) implementieren.

Die übrigen Bausteine eines Experiments hängen an dieser Kette:

- [config_reader/](config_reader/)
  - liest die YAML-Experimentkonfiguration ein
  - wandelt die YAML-Abschnitte in passende Dataclasses um

- [src/pipelines/](src/pipelines/)
  - verbindet die Konfiguration mit Feature-Extraktion, Clustering, Evaluation und Interpretation
  - baut daraus den konkreten Ausführungsablauf des Experiments

- [src/experiments/base.py](src/experiments/base.py)
  - stellt mit `BaseExperiment` den allgemeinen `run()`-Ablauf bereit
  - nutzt die Hooks `load_config()`, `build_pipeline()` und `save_results()` für experiment-spezifische Anpassungen

Weitere Ordner wie [data/raw/](data/raw/), [experiments/](experiments/), [src/clustering/](src/clustering/), [src/features/](src/features/), [src/evaluation/](src/evaluation/) und [src/interpretation/](src/interpretation/) liefern die jeweiligen Daten und Implementierungen für diese Kette.

Die Trennung ist bewusst so gewählt, dass experiment-spezifischer Code klein bleibt und wiederverwendbare Logik an zentraler Stelle liegt.

## 2. Anlegen eines Experiments

Ein neues Experiment wird als eigener Ordner unter [experiments/](experiments/) angelegt, zum Beispiel `kmeans_tfidf/`.

Ein typischer Experimentordner enthält:

- eine ausführbare Python-Datei, zum Beispiel `kmeans_tfidf.py`
- eine YAML-Datei mit der Konfiguration, zum Beispiel `kmeans_tfidf.yaml`
- einen oder mehrere Ergebnisordner vom Typ `results_*/` die von den Runs automatisch erstellt werden

Die YAML-Datei beschreibt die komplette Experimentausführung. In ihr werden typischerweise Angaben für folgende Bereiche gespeichert:

- Input
- Features
- Clustering
- Output
- Interpretation

Die Python-Datei im Experimentordner ist der Einstiegspunkt. Sie lädt die Konfiguration über den passenden ConfigReader, baut die Pipeline zusammen und startet den Ablauf.

Der technische Ablauf eines Experiments ist in der Regel:

| Schritt | Ebene | Komponente | Aufgabe |
| --- | --- | --- | --- |
| 1 | Experiment + Config | `BaseExperiment.load_config()` / ConfigReader | YAML laden, validieren und in Dataclasses umwandeln |
| 2 | Experiment + Input | `BaseExperiment.load_documents()` | Eingabedokumente aus [data/raw/](data/raw/) anhand der geladenen Input-Config einlesen |
| 3 | Pipeline | [src/features/](src/features/) | Feature-Extraktion ausführen |
| 4 | Pipeline | [src/clustering/](src/clustering/) | Clustering anwenden |
| 5 | Pipeline | [src/evaluation/](src/evaluation/) | Metriken und Evaluation berechnen |
| 6 | Pipeline | [src/interpretation/](src/interpretation/) | Cluster interpretieren |
| 7 | Experiment | `save_results()` | Ergebnisse in `results_*/` speichern |

## 3. Wie man einen ConfigReader hinzufügt, wenn man muss

Die Notwendigkeit für einen neuen ConfigReader entsteht immer dann, wenn ein Experiment eine neue Konfiguration benötigt oder eine bestehende YAML-Struktur erweitert wird. Der Ablauf ist in diesem Projekt bewusst klar getrennt: Zuerst wird die Dataclass definiert, dann der passende Section-Reader implementiert und danach der Reader im Builder registriert.

Der typische Flow sieht am Beispiel von `kmeans` so aus:

1. Zuerst wird eine neue Config-Dataclass erstellt, zum Beispiel `KMeansConfig` in [src/clustering/kmeans.py](src/clustering/kmeans.py).
2. Diese Dataclass bildet genau die Felder ab, die in der YAML-Datei für den Algorithmus gesetzt werden können oder müssen.
3. Danach wird ein neuer Reader angelegt, der `ConfigSectionReader` implementiert.
4. Dieser Reader liest nur die zuständige YAML-Sektion, validiert sie und erzeugt daraus die passende Dataclass.
5. Anschließend wird der Reader in [config_reader/config_reader_new.py](config_reader/config_reader_new.py) über das Builder-Pattern registriert, im Beispiel mit `add_kmeans()`.
6. Ab diesem Punkt kann sich das Experiment seinen benötigten ConfigReader über den Builder zusammensetzen.
7. Das Experiment nutzt dann den zusammengesetzten Reader, um die YAML zu laden und die Konfiguration als Dataclasses an die Pipeline weiterzugeben.

Ein ConfigReader übernimmt damit vor allem folgende Aufgaben:

- Laden der YAML-Datei
- Prüfen, ob die Pflichtfelder vorhanden sind
- Aufteilen in logische Konfigurationsbereiche wie `input`, `tfidf`, `kmeans`, `dbscan`, `optics`, `hdbscan`, `agglomerative`, `affinityPropagation`, `spectral`, `gaussianMixture`, `interpretation`, `interpretation_bert` und `outputs`
- Erzeugen der passenden Dataclasses für die jeweiligen Abschnitte
- Bereitstellen einer einheitlichen Schnittstelle für die Pipeline

Die Section-Reader sollten dabei möglichst streng nur eine Aufgabe erfüllen: YAML-Mapping einlesen, validieren und in die zugehörige Dataclass umwandeln. 

Wichtig ist dabei, dass die Dataclass immer die YAML-Struktur widerspiegelt: Jede Konfigurationsoption, die die YAML befüllen muss, sollte als Feld, Typ und Validierungslogik in der Dataclass und im Reader sichtbar sein. So bleibt die Struktur konsistent und neue Experimente können dieselbe Logik wiederverwenden.

## 4. Die Pipeline der Experimente

#### Ziel

- Die Pipeline ist die Ausführungs-Engine: sie kapselt Feature-Extraktion, Optimierung, Evaluation und optionale Interpretation. Die YAML-Konfiguration wählt Komponenten; die Pipeline führt sie standardisiert aus.

#### Standard-Ablauf der Pipeline (kompakt, mit eingerückten Unterpunkten)

- Eingabe / Vorbedingung (wichtig: findet *nicht* in der Pipeline statt)
  - Die Pipeline erwartet als Eingabe eine Liste von `Document`-Objekten (Text + optional `doi`).
  - Das Einlesen der YAML, das Erzeugen eines `CombinedConfig` und das Erstellen der `Document`-Objekte werden vorher ausgeführt (z. B. in `BaseExperiment.load_config()` und `BaseExperiment.load_documents()` mittels des `ConfigReader`).

- Feature-Extraktion (einmal pro Lauf)
  - Der konkrete `FeatureExtractor` wird (von der Factory) erzeugt und auf die Texte angewendet.

- Optimierung / Trials (Optuna)
  - Die Pipeline startet eine Optuna-Schleife über die für den Clusterer definierten Suchräume.
  - Pro Trial: Clusterer-Variante erzeugen → `fit_predict` auf den Features → Evaluierung.
  - Jede Trial-Auswertung wird als `RunSummary` protokolliert.

- Auswahl, Interpretation, Ausgabe
  - Auswahl des besten Trials (mehrkriteriell).
  - Optional: Interpreter (von der Factory) erzeugen und auf den besten Lauf anwenden.
  - Erzeuge `document_cluster_mapping` und setze Metadaten für die Ergebnisdateien.

(Die Implementierung des Ablaufes ist in [src/pipelines/pipeline.py](../src/pipelines/pipeline.py#L1-L200) dokumentiert.)

#### Rolle des Pipeline-Factory/Builders

- `PipelineBuilder` (Factory) nimmt ein `CombinedConfig` und erzeugt ein `PipelineConfig` mit den konkreten Komponenten: `feature_extractor`, `clusterer_factory`, `clusterer_name`, `clusterer_config`, `evaluator`, `interpreter` und `metadata`.
- Autodetection: Der Builder prüft die gesetzten Sektionen im `CombinedConfig` (z. B. `bert`, `tfidf`, `kmeans`, `hdbscan`) und ruft die passenden Factories auf (`FeatureExtractorFactory`, `ClustererFactory`, `InterpreterFactory`).
- Injection: Evaluator (z. B. `BasicUnsupervisedEvaluator`) und Metadaten (z. B. `experiment_name`) werden in das `PipelineConfig` injiziert.

#### Trennung von Verantwortung

- Konfiguration (YAML & `ConfigReader`) — Was der Nutzer wählt.
- Pipeline — Wie es ausgeführt wird (optimiert, evaluiert, interpretiert).
- Änderungen an Optimierungslogik, Evaluator oder Trial-Ablauf erfolgen zentral in der Pipeline und betreffen automatisch alle Experimente.

#### Integration (kurz)

- `BaseExperiment.build_pipeline()` verwendet den `PipelineBuilder`/Factory; so müssen konkrete Experiment-Module nur `load_config()` und `save_results()` implementieren (siehe [src/experiments/base.py](../src/experiments/base.py#L1-L200)).
- Hinweis: Das Parsen der YAML und das Erzeugen der `Document`-Eingabe erfolgen vor dem Aufruf der Pipeline (z. B. in `BaseExperiment.load_config()` / `BaseExperiment.load_documents()`), die Pipeline selbst behandelt Feature-Extraktion, Optimierung, Evaluation und Interpretation.

#### Referenzen

- Pipeline-Implementation: [src/pipelines/pipeline.py](../src/pipelines/pipeline.py#L1-L200)
- Factory/Builder: [src/pipelines/pipeline_builder.py](../src/pipelines/pipeline_builder.py#L1-L200)

## 5. Zusammenspiel der Hauptverzeichnisse

Die Architektur des Projekts folgt einer klaren Zuständigkeit:

- [config_reader/](config_reader/) ist für das Einlesen der Konfiguration zuständig
- [experiments/](experiments/) definiert konkrete Experimente und ihre Ausführung
- [src/](src/) enthält die eigentliche Implementierungslogik
- [data/raw/](data/raw/) liefert die Datenbasis
