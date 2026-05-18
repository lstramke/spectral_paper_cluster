# Hinweise zur Benutzung

Im Folgenden sind Hinweise zur standardmäßigen Benutzung und Änderungen, die man vornehmen kann ohne Code zu verändern, aufgeführt.

## Datensätze

Datensätze müssen als csv im data/raw Ordner abgelegt werden. Dabei muss folgendes Format eingehalten werden:

```
title;abstract;doi
<Beispiel_Titel>;<Beispiel_Abstract>;<Beispiel_DOI>
...
```

,die DOI ist dabei optional.

Zur Umstellung der Experimente auf den Datensatz muss in dem zugehörigen Config File, die yaml Datei im spezifischen Experimentordner, der Input Pfad geändert werden.

Beispiel: 
Du hast einen neuen Datensatz toller_neuer_datensatz.csv angelegt und möchtest jetzt das affinityPropagation_bert Experiment auf diesen Datensatz ausführen.
Führe dafür folgende Veränderungen in `experiments/affinityPropagation_bert/affinityPropagation_bert.yaml` durch:

```yaml
experiment_name: affinityPropagation_bert_2086 -> affinityPropagation_bert_toller_neuer_Datensatz

input:
  documents_path: data/raw/dataset_2086.csv -> data/raw/toller_neuer_datensatz.csv
  format: csv
  text_fields: [title, abstract, doi]
  fuse_mode: join
  separator: ";"
```

Diese Änderungen kannst du direkt an der Datei vornehmen oder über die cluster_cli in der Option `Edit config` > `affinityPropagation_bert` > `input` dann das Feld `documents_path` sowie unter `Edit config` > `affinityPropagation_bert` > `experiment_name`.

## Automatische Auswertungsdateien

Für jeden erfolgreichen Experimentdurchlauf werden ein png Plot, ein interaktiver html Plot und json Dateien für die Run Ergebnisse erzeugt. Das zusammenfassende Markdown Dokument wird nicht automatisch erzeugt.

## Interaktive Werkzeuge

-- **Propagate labels:** Aufruf über das interaktive CLI (`python cluster_cli.py`) → Menüpunkt `Propagate labels`. Du wählst ein Experiment und bestätigst Eingabedateien sowie Optionen interaktiv. Das Ergebnis wird in `data/labels/processed/<experiment>_propagated_labels.csv` abgelegt.

-- **Review rules / Regelerweiterung:** Aufruf über das interaktive CLI (`python cluster_cli.py`) → Menüpunkt `Review rules`. Das Tool zeigt Regelvorschläge an, die Du interaktiv prüfen und annehmen oder verwerfen kannst. Neu erzeugte oder geänderte Regeln werden ggf. unter `data/rules/processed/` gespeichert.

## Weiterverwendung der erzeugten Dateien

- **Regeldateien (`data/rules/processed/`)**: Die neuen oder erweiterten Regeldateien können als neue Basisi für den Regelklassifizierer verwendet werden. So lassen sich Regeln versioniert testen oder selektiv in Auswertungs-/Produktionsläufe einbinden.

- **Propagierte Labels (`data/labels/processed/<experiment>_propagated_labels.csv`)**: Die erzeugten Label-CSV-Dateien eignen sich als Trainings- oder Fine-Tuning-Datensatz für überwachte ML-Modelle (z. B. Textklassifikatoren) oder als zusätzliches Label-Set in semi-supervised Ansätzen. Sie können auch zur Evaluierung oder für Active-Learning-Workflows verwendet werden. Beachte: Die Qualität der propagierten Labels ist nicht gleichzusetzen mit vollständig manueller Labelung; überprüfe und ggf. korrigiere die Labels, bevor Du sie für Trainingszwecke verwendest.
