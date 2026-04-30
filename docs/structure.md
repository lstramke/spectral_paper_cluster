# Gesamtstruktur des Projekts

Das Projekt ist so aufgebaut, dass Experimente möglichst direkt aus der Ordnerstruktur heraus nachvollziehbar und erweiterbar sind. Die zentrale Trennung liegt zwischen Konfiguration, Daten, Experiment-Definitionen und der eigentlichen Implementierungslogik in `src/`.

## 1. Komponenten eines Experiments

Ein Experiment besteht nicht nur aus dem Skript im jeweiligen Ordner, sondern aus mehreren klar getrennten Bausteinen. Im Zentrum steht dabei `BaseExperiment` aus [src/experiments/base.py](src/experiments/base.py): Die Klasse stellt mit `run()` einen allgemeinen Ablauf bereit und kapselt die gemeinsamen Schritte für alle Experimente.

Die konkrete Ausführung wird über drei Hooks gesteuert:

- `load_config()`
  - baut den passenden ConfigReader über das Builder-Pattern zusammen
  - lädt und validiert die YAML-Konfiguration mit dem ConfigReader
  - setzt die experiment-spezifische Config-Dataclass

- `build_pipeline()`
  - erzeugt die passende Pipeline-Klasse
  - parametrisiert die Pipeline mit der geladenen Konfiguration

- `save_results()`
  - bestimmt, wie Ausgaben, JSON-Dateien, Plots oder Summarys gespeichert werden
  - kann experimentabhängig frei angepasst werden

`BaseExperiment.run()` nutzt diese Hooks dann für den allgemeinen Ablauf: Konfiguration laden, Dokumente einlesen, Pipeline bauen, Pipeline ausführen und Ergebnisse speichern. Dadurch bleibt die gemeinsame Logik zentral, während einzelne Experimente nur ihre eigenen Details flexibel implementieren.

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

## 4. Zusammenspiel der Hauptverzeichnisse

Die Architektur des Projekts folgt einer klaren Zuständigkeit:

- [config_reader/](config_reader/) ist für das Einlesen der Konfiguration zuständig
- [experiments/](experiments/) definiert konkrete Experimente und ihre Ausführung
- [src/](src/) enthält die eigentliche Implementierungslogik
- [data/raw/](data/raw/) liefert die Datenbasis
