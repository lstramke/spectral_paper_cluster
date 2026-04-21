# gaussianMixture + tfidf auf 41

## Kurzüberblick

- **Kurzbeschreibung:** kurze, natürliche Beschreibung des Experiments und der Zielsetzung.

- **Kurzbeschreibung:** Unüberwachtes Clustering von Dokumenten mithilfe von TF‑IDF (mit optionaler LSA) gefolgt von einem Gaussian Mixture Model (GMM). Ziel ist es, semantische Gruppen im Korpus zu identifizieren und die Clusterqualität mit etablierten Metriken zu bewerten.

## Konfiguration

Die Experimentkonfiguration liegt in [gaussianMixture_tfidf.yaml](../gaussianMixture_tfidf.yaml).

```yaml
experiment_name: gaussianMixture_tfidf

input:
  documents_path: data/raw/data_db_raw.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

gaussianMixture:
  n_components: 10
  tol: 0.001
  reg_covar: 1e-6
  max_iter: 200
  n_init: 5
  init_params: k-means++
  random_state: 42
  covariance_type: full


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
  output_dir: experiments/gaussianMixture_tfidf/results_41
  plot_name: gaussianMixture_tfidf_pca.png
  summary_name: best_gaussianMixture_tfidf_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py`
3. Clustering mit `src/clustering/gaussianMixture.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner `results_41/` speichern

## Ergebnisse

### Plot:

![GaussianMixture + tfidf PCA](gaussianMixture_tfidf_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [gaussianMixture_tfidf_pca.html](gaussianMixture_tfidf_pca.html)

### Metriken: 

Die in der JSON gespeicherten Kennzahlen direkt auswerten.

Die Metriken werden in `best_gaussianMixture_tfidf_summary.json` gespeichert. Für das aktuelle Experiment ergeben sich folgende Werte:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.1031 | sehr gering — schwache Trennung |
| Davies–Bouldin Index | 2.0178 | mittel bis hoch — deutliche Überlappung |
| Calinski–Harabasz Index | 1.9930 | niedrig (relativ zur Datenmenge) |

### Cluster-Interpretation
Für die Interpretation wurden die Top‑Wörter aus dem nicht reduzierten TF‑IDF‑Raum verwendet; die zugehörigen Gewichte finden sich in `best_gaussianMixture_tfidf_summary.json`.

| Cluster | Top-Wörter |
| ---: | --- |
| 0 | vision, spectroscopy, technologies, technology, based, capabilities, use, spatial, different, monitoring |
| 1 | patients, studies, detection, small, meta analysis, meta, lesions, improvement, results, literature |
| 2 | tissue, technology, information, brain, biological, provides, recent, diagnosis, data, disease |
| 3 | cancer, accuracy, sensitivity, aided, computer aided, specificity, high, detection, computer, techniques |
| 4 | medical, medical applications, research, challenges, clinical, limitations, field, future, study, technology |
| 5 | tissue, guidance, systems, vivo, surgical, patients, current, studies, modality, different |
| 6 | perfusion, clinical, light, surgery, measurements, results, literature, color, gastrointestinal, promising |
| 7 | disease, disorders, field, current, clinical, brain, early, approaches, diseases, significant |
| 8 | multispectral, lesions, skin, multispectral imaging, systematic, summarize, level, systematic review, cancer, hyperspectral multispectral |
| 9 | learning, medical, images, biological, image, algorithms, data, techniques, various, machine |

## Evaluation

Die Kennzahlen zeigen eine insgesamt schwache Clusterstruktur: der Silhouette‑Score (0.1031) spricht für geringe Trennung, der Davies–Bouldin‑Index (2.0178) weist auf  Überlappungen hin und der Calinski–Harabasz‑Index ist vergleichsweise niedrig. Praktische nächste Schritte sind:

- `n_components` feiner abstufen (z. B. Grid- oder Elbow‑Suche)
- andere `covariance_type`-Einstellungen testen (`tied`, `diag`, `spherical`) und `n_init` erhöhen
- TF‑IDF‑Preprocessing anpassen (Stopwords, `min_df`, LSA‑Dimensionen)
