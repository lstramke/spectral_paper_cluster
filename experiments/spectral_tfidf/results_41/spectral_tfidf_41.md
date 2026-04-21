# spectral + tfidf auf 41

## Kurzüberblick

- **Kurzbeschreibung:** TF‑IDF‑Vektoren (optional LSA) werden über einen kNN‑Affinitätsgraph (Cosine) oder RBF‑Kernel an Spectral Clustering übergeben, um thematische Dokumentengruppen — auch nicht‑konvexe Strukturen — zu entdecken; geeignet für mittelgroße Datensätze.

## Konfiguration

Die Experimentkonfiguration muss in [spectral_tfidf.yaml](../spectral_tfidf.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: spectral_tfidf

input:
  documents_path: data/raw/data_db_raw.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

spectral:
  n_clusters: 10
  affinity: nearest_neighbors
  eigen_solver: arpack
  assign_labels: kmeans
  n_init: 10
  gamma: 1.0
  n_neighbors: 10
  random_state: 42
  n_jobs: 1


tfidf:
  max_features: 1000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.5
  lowercase: true
  stop_words: english
  extra_stop_words: ["hsi"]
  use_lsa: true
  lsa_components: 40

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/spectral_tfidf/results_41
  plot_name: spectral_tfidf_pca.png
  summary_name: best_spectral_tfidf_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py`
3. Clustering mit `src/clustering/spectralClustering.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner unter `results_41/` speichern

## Ergebnisse

### Plot:

![spectral + tfidf PCA](spectral_tfidf_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [spectral_tfidf/spectral_tfidf_pca.html](spectral_tfidf_pca.html)

### Metriken:

Die Metriken werden in `best_spectral_tfidf_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.11248621344566345 | schwache bis mäßige Trennung |
| Davies–Bouldin Index | 1.9227023029672683 | mittlere Überlappung |
| Calinski–Harabasz Index | 2.0304430584237467 | schwache Clusterstruktur |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster (Top‑10), berechnet aus den nicht reduzierten TF‑IDF‑Features:

| Cluster | Top‑Wörter |
| --- | --- |
| 0 | multispectral, vision, capabilities, spectroscopy, technologies, use, using, limitations, different, monitoring |
| 1 | medical, learning, algorithms, research, medical applications, images, future, study, techniques, machine |
| 2 | studies, vivo, patients, measurements, tissue, detection, sensitivity, performed, systems, literature |
| 3 | cancer, skin, color, computer, computer aided, aided, accuracy, skin cancer, diagnostic, light |
| 4 | technology, information, data, recent, diagnosis, provides, disease, diseases, medical, spatial |
| 5 | tissue, high, brain, guidance, different, used, resolution, types, modality, surgical |
| 6 | clinical, perfusion, surgery, gastrointestinal, promising, results, spectral imaging, main, systems, article |
| 7 | lesions, patients, skin, multispectral, small, multispectral imaging, significant, level, advances, tissue |
| 8 | disease, disorders, field, current, clinical, brain, early, approaches, diseases, significant |
| 9 | biological, proposed, images, tissue, tissues, use, related, resolution, visualization, image |

## Evaluation

Die aktuelle Konfiguration liefert nur eine schwache bis mäßige Trennung (Silhouette ≈ 0.11) bei moderater Überlappung (Davies–Bouldin ≈ 1.92); die Clusterstruktur ist insgesamt eher schwach. Empfehlung: `n_neighbors`, `gamma` und `affinity` (kNN vs. rbf/precomputed) feinabstimmen.
