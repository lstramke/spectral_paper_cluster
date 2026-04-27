# dbscan + fasttext auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** Dokumente werden in Fasttext-Embeddings überführt (TruncatedSVD zur weiteren Dimesnionsreduktion) gefolgt von DBSCAN-Clustering. Ziel ist, dichte, thematische Gruppen ohne feste Clusteranzahl zu identifizieren.

## Konfiguration

Die Experimentkonfiguration muss in [dbscan_fasttext.yaml](../dbscan_fasttext.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: dbscan_fasttext_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

dbscan:
  eps_range: [0.1, 1.0]
  min_samples_range: [2, 20]
  n_trials: 400
  metric: cosine
  leaf_size: 30
  p:
  n_jobs:

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/dbscan_fasttext/results_2086
  plot_name: dbscan_fasttext_2086_pca.png
  summary_name: best_dbscan_fasttext_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/fasttext.py`
3. Clustering mit `src/clustering/dbscan.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: PNG-Plot und Summary-JSON im Unterordner `results_2086/` speichern

## Ergebnisse

### Plot:

![dbscan + fasttext PCA](dbscan_fasttext_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [dbscan_fasttext_2086_pca.html](dbscan_fasttext_2086_pca.html)

### Metriken:

Die Metriken werden in `best_dbscan_fasttext_2086_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.892043948173523 | |
| Davies–Bouldin Index | 0.3388814357842144 | |
| Calinski–Harabasz Index | 16.66710516056775 | |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF-Raum; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung.

| Cluster | Top-Wörter |
| --- | --- |
| -1 | 006, 007, 001, 005, 002, 0003, 000, 00, 0001, 003 |
| 0 | segmentation, raman, photoacoustic, perfusion, brain, fusion, lesions, 3d, optoacoustic, nir |

## Evaluation
Metriken sehr gut, Ergebnis nicht nutzbar.

"cluster_sizes": {
  "-1": 2,
  "0": 2084
}