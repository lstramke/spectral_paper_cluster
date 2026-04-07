# Experiments

Zusammenfassung und Auswertung der durchgeführten Clustering-Experimente.

## kmeans + tfidf

### Kurzüberblick

- **Kurzbeschreibung:** Texte werden in TF‑IDF‑Vektoren umgewandeln und per `k-means` gruppiert, um sinnvolle, gut interpretierbare Cluster (z. B. Themen oder Dokumentengruppen) zu finden. Ziel ist es, aus den Clustern verwertbare Einsichten zu gewinnen.

### Konfiguration

Die Experimentkonfiguration liegt in [configs/kmeans_tfidf.yaml](configs/kmeans_tfidf.yaml).

```yaml
experiment_name: kmeans_tfidf

input:
  documents_path: data/raw/data_db_raw.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ","

pipeline:
  n_clusters: 8
  max_iter: 100
  tol: 0.0001
  seed: 42

tfidf:
  max_features: 2000
  ngram_range: [1, 2]
  min_df: 4
  max_df: 0.75
  lowercase: true
  use_lsa: false
  lsa_components: 100

outputs:
  output_dir: outputs/kmeans_tfidf
  plot_name: kmeans_tfidf_pca.png
  summary_name: kmeans_tfidf_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py` (TF‑IDF, optional LSA)
3. `k-means` Clustering (siehe `src/clustering/kmeans.py`)
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: PCA wird zur 2D-Visualisierung nach dem Clustering angewendet. Plot und Metrik-JSON werden zusammen in einem Unterordner unter `outputs/` abgelegt.

### Ergebnisse

Das Ergebnisbild und die zugehörige JSON-Zusammenfassung werden im Experiment-Unterordner unter `outputs/` abgelegt (z. B. `outputs/kmeans_tfidf/`).

#### Plot (PCA):

![kmeans + tfidf PCA](outputs/kmeans_tfidf/kmeans_tfidf_pca.png)

#### Metriken:

Die Metriken werden in `outputs/kmeans_tfidf/kmeans_tfidf_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.0143 | sehr niedrig, Cluster sind nur schwach getrennt |
| Davies–Bouldin Index | 2.6667 | eher hoch, Cluster überlappen noch deutlich |
| Calinski–Harabasz Index | 1.5056 | sehr niedrig, geringe Clusterstruktur |

### Evaluation

Die JSON zeigt, dass das aktuelle Setup nur eine schwache Clusterstruktur liefert. Der Silhouette Score ist nahe 0, der Davies–Bouldin Index ist relativ hoch und auch der Calinski–Harabasz Index ist sehr klein. Das spricht dafür, dass die Parametrisierung noch nicht zu klar getrennten, inhaltlich starken Clustern führt.

---

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


#### Metriken: Die in der JSON gespeicherten Kennzahlen direkt auswerten.

Die Metriken werden in `outputs/<experiment>/<experiment>_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | <value> | <kurze Bewertung> |
| Davies–Bouldin Index | <value> | <kurze Bewertung> |
| Calinski–Harabasz Index | <value> | <kurze Bewertung> |

### Evaluation

Kurze Interpretation der Ergebnisse und mögliche nächste Schritte. Hier kann auch direkt auf die Summary-JSON verwiesen werden.
