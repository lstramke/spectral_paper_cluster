# hdbscan + tfidf auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** TF‑IDF‑Feature‑Extraktion (optional LSA) gefolgt von HDBSCAN‑Clustering; HDBSCAN extrahiert stabile dichtebasierte Cluster ohne globales eps und liefert außerdem Cluster‑Stabilitäten und probabilistische Mitgliedschaften. Ziel ist die explorative Identifikation thematischer Gruppen und robustes Rauschen‑Handling.

## Konfiguration

Die Experimentkonfiguration muss in [hdbscan_tfidf.yaml](../hdbscan_tfidf.yaml) einegtragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: hdbscan_tfidf_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

hdbscan:
  min_cluster_size_range: [5, 30]
  min_samples_range: [1, 5]
  metric: euclidean
  cluster_selection_method: eom
  n_trials: 400

tfidf:
  max_features: 5000
  ngram_range: [1, 2]
  min_df: 0.001
  max_df: 0.09
  lowercase: true
  stop_words: english
  extra_stop_words: ["hsi"]
  use_lsa: true
  lsa_components: 100

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/hdbscan_tfidf/results_2086
  plot_name: hdbscan_tfidf_2086_pca.png
  summary_name: best_hdbscan_tfidf_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py`
3. Clustering mit `src/clustering/hdbscan.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner `results_2086/` speichern

## Ergebnisse

### Plot:

![hdbscan + tfidf PCA](hdbscan_tfidf_2086_pca.png)


Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [hdbscan_tfidf_pca.html](hdbscan_tfidf_2086_pca.html)


### Metriken:

Die Metriken werden in `best_hdbscan_tfidf_2086_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.08024097979068756 |  |
| Davies–Bouldin Index | 2.1040121387844115 |  |
| Calinski–Harabasz Index | 14.443256265588882 | |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF‑Raum; die zugehörigen Gewichte stehen in `best_hdbscan_tfidf_2086_summary.json`.

| Cluster | Top-Wörter |
| ---: | --- |

| Cluster | Top-Wörter |
| ---: | --- |
| -1 | perfusion, 3d, oxygen, photoacoustic, oxygenation, deep learning, nir, saturation, intraoperative, noise |
| 0 | cervical, cervical cancer, colposcopy, cin, mdc, cervix, colposcope, neoplasia, acetic, acetic acid |
| 1 | tongue, tongue diagnosis, medicine, coating, tongue images, tongue color, kampo, tcm, segmentation, chinese medicine |
| 2 | laparoscopic, laparoscope, video, cm, broom, push broom, push, intraoperative, minimally invasive, distances |
| 3 | endoscopy, comb, flexible, msi, endoscope, endoscopic, cubes, pattern, tract, modalities |
| 4 | bacterial, pathogens, optica, species, publishing, bacteria, group terms, biofilms, aureus, optica publishing |
| 5 | thyroid, nodules, thyroid nodules, cerenkov, needle, single cells, hyperspectral raman, benign, raman, fine |
| 6 | caries, enamel, dental, surfaces, teeth, occlusal, dental caries, surface, tooth, swir |
| 7 | carcinoma, cell carcinoma, scc, squamous, actinic, squamous cell, keratosis, actinic keratosis, skin cancer, carcinomas |
| 8 | melanoma, lesions, melanomas, pigmented, lesion, dermoscopy, benign, nevi, specificity, malignant |
| … | weitere 63 Cluster (siehe `best_hdbscan_tfidf_2086_summary.json`) |

## Evaluation
