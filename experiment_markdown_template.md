## Template für weitere Experimente

## <experiment_name>

### Kurzüberblick

- **Kurzbeschreibung:** kurze, natürliche Beschreibung des Experiments und der Zielsetzung.

### Konfiguration

Die Experimentkonfiguration liegt in [configs/<experiment>.yaml]().

```yaml
experiment_name: <experiment_name>

file body
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/<feature_extractor>.py`
3. Clustering mit `src/clustering/<algorithm>.py`
4. Evaluation mit `src/evaluation/<evaluator>.py`
5. Outputs: Plot und Summary im Unterordner unter `outputs/` speichern

### Ergebnisse

#### Plot:

![<experiment>]()

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [outputs/<experiment>/<experiment>_pca.html]()

#### Metriken: 

Die in der JSON gespeicherten Kennzahlen direkt auswerten.

Die Metriken werden in `outputs/<experiment>/<experiment>_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | <value> | <kurze Bewertung> |
| Davies–Bouldin Index | <value> | <kurze Bewertung> |
| Calinski–Harabasz Index | <value> | <kurze Bewertung> |

#### Cluster-Interpretation

### Evaluation

Kurze Interpretation der Ergebnisse und mögliche nächste Schritte. Hier kann auch direkt auf die Summary-JSON verwiesen werden.