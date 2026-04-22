# agglomerative + tfidf auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** TF‑IDF‑Vektoren (mit optionaler LSA‑Reduktion) werden verwendet, um Dokumente über kosinus‑basierte Ähnlichkeit zu gruppieren. Die agglomerative Clusterung schneidet den Dendrogramm‑Baum über einen Distanz‑Threshold, sodass thematisch ähnliche Dokumentgruppen extrahiert werden können. Ziel ist die explorative Identifikation von Themen und die anschließende Interpretierbarkeit der Cluster.

## Konfiguration

Die Experimentkonfiguration muss in [agglomerative_tfidf.yaml](../agglomerative_tfidf.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: agglomerative_tfidf_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

agglomerative:
  distance_threshold_range: [0.1, 1.0]
  n_trials: 1000
  metric: cosine
  linkage: average
  compute_full_tree: true

tfidf:
  max_features: 1000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.5
  lowercase: true
  stop_words: english
  extra_stop_words: ["hsi"]
  use_lsa: true
  lsa_components: 100

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/agglomerative_tfidf/results_2086
  plot_name: agglomerative_tfidf_2086_pca.png
  summary_name: best_agglomerative_tfidf_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py`
3. Clustering mit `src/clustering/agglomerativeClustering.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner `results_2086/` speichern

### Ergebnisse

#### Plot:

![agglomerative + tfidf PCA](agglomerative_tfidf_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [agglomerative_tfidf_pca.html](agglomerative_tfidf_pca.html)

#### Metriken:

Die Metriken werden in `best_agglomerative_tfidf_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.18110454082489014  |  |
| Davies–Bouldin Index | 1.263998561526703 |  |
| Calinski–Harabasz Index | 6.369068977866156 |  |

#### Cluster-Interpretation

| Cluster | Top‑Wörter |
| --- | --- |
| 0 | learning, deep, deep learning, cancer, attention, data, medical, framework, domain, feature |
| 1 | band, ratio, selection, narrow, contrast, tissue, dual, proposed, prediction, imaging techniques |
| 2 | unmixing, linear, end, pixel, non, matrix, negative, method, algorithm, nonlinear |
| 3 | cameras, information, camera, multispectral imaging, device, medical, used, light, applications, monitoring |
| 4 | swir, short, wave, infrared, hyperspectral imaging, nm, validation, collagen, phantoms, analysis |
| 5 | data, image data, software, analysis, sets, medical image, processing, tools, visible, spectroscopic |
| 6 | skin, line, illumination, laser, rgb, maps, mapping, snapshot, data, nm |
| 7 | systems, devices, sensing, applications, optical, spectral imaging, sensors, advanced, integration, design |
| 8 | calibration, scanning, hyperspectral imaging, applications, video, biomedical, spatial, device, custom, acquisition |
| 9 | classification, deep, learning, deep learning, network, medical, hyperspectral image, proposed, accuracy, classify |
| … | weitere 534 Cluster (siehe `best_agglomerative_tfidf_2086_summary.json`) |

### Evaluation

