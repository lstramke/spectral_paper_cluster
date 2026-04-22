## Template für weitere Experimente

# <experiment_name> auf <dataset>

## Kurzüberblick

- **Kurzbeschreibung:** kurze, natürliche Beschreibung des Experiments und der Zielsetzung.

## Konfiguration

Die Experimentkonfiguration muss in [<experiment>.yaml]() eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: <experiment_name>

file body
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/<feature_extractor>.py`
3. Clustering mit `src/clustering/<algorithm>.py`
4. Evaluation mit `src/evaluation/<evaluator>.py`
5. Outputs: Plot und Summary im Unterordner `outputs/` speichern

## Ergebnisse

### Plot:

![<experiment>]()

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [outputs/<experiment>_pca.html]()

### Metriken: 

Die in der JSON gespeicherten Kennzahlen direkt auswerten.

Die Metriken werden in `outputs/<experiment>_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | <value> | <kurze Bewertung> |
| Davies–Bouldin Index | <value> | <kurze Bewertung> |
| Calinski–Harabasz Index | <value> | <kurze Bewertung> |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF‑Raum; die zugehörigen Gewichte stehen in `outputs/best_<experiment>_summary.json`.

| Cluster | Top-Wörter |
| --- | --- |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |

## Evaluation

Kurze Interpretation der Ergebnisse und mögliche nächste Schritte. Hier kann auch direkt auf die Summary-JSON verwiesen werden.