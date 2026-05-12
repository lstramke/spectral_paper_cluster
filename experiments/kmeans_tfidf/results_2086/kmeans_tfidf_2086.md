# kmeans + tfidf auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** Texte werden in TF‑IDF‑Vektoren umgewandeln und per `k-means` gruppiert, um sinnvolle, gut interpretierbare Cluster (z. B. Themen oder Dokumentengruppen) zu finden. Ziel ist es, aus den Clustern verwertbare Einsichten zu gewinnen.

## Konfiguration

Die Experimentkonfiguration muss in [kmeans_tfidf.yaml](../kmeans_tfidf.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: kmeans_tfidf_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

kmeans:
  cluster_range: [5, 40]
  max_iter: 400
  tol: 0.00001
  seed_range: [1, 10000]
  n_trials: 1200

tfidf:
  max_features: 5000
  ngram_range: [1, 2]
  min_df: 0.001
  max_df: 0.09
  lowercase: true
  stop_words: english
  extra_stop_words: ["don", "like", "hsi"]
  use_lsa: true
  lsa_components: 100

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/kmeans_tfidf/results_2086
  plot_name: kmeans_tfidf_2086_pca.png
  summary_name: best_kmeans_tfidf_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py` (TF‑IDF, optional LSA)
3. `k-means` Clustering (siehe `src/clustering/kmeans.py`)
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: PCA wird zur 3D-Visualisierung nach dem Clustering angewendet. Plot und Metrik-JSON werden zusammen in einem Unterordner `results_2086/` abgelegt.

## Ergebnisse

Das Ergebnisbild und die zugehörige JSON-Zusammenfassung werden im Experiment-Unterordner unter `results_2086/` abgelegt.

### Plot (PCA):

![kmeans + tfidf PCA](kmeans_tfidf_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [kmeans_tfidf_2086_pca.html](kmeans_tfidf_2086_pca.html)

### Metriken:

Die Metriken für alle Zufallswerte werden in [`kmeans_tfidf_2086_all_runs.json`](kmeans_tfidf_2086_all_runs.json) gespeichert. Die Details zum besten Lauf stehen zusätzlich in [`best_kmeans_tfidf_2086_summary.json`](best_kmeans_tfidf_2086_summary.json). Für den aktuellen besten Lauf ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.20120152831077576 | bis mäßig |
| Davies–Bouldin Index | 2.650162590399797 | groß |
| Calinski–Harabasz Index | 26.14920387719426 | in Ordnung |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF-Raum; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung. Es wurde die Gruppierung des besten Seeds interpretiert.

| Cluster | Top-Wörter |
| --- | --- |
| 0 | nanoparticles, nps, gold, gold nanoparticles, nanoparticle, particle, np, particles, biomedical applications, msot |
| 1 | perfusion, flap, oxygenation, index, tissue oxygenation, sto, nir, twi, thi, hemoglobin |
| 2 | remote, remote sensing, sensing, review, deep learning, vision, technologies, image processing, unmixing, article |
| 3 | brain, brain tumor, brain cancer, cerebral, segmentation, human brain, brain tumors, tumors, mapping, neurosurgical |
| 4 | autofluorescence, fluorescent, mice, protein, excitation, fluorescence imaging, species, cellular, drug, stem |
| 5 | mu, polarization, thz, scattering, terahertz, mid, photodetectors, broadband, oct, absorption |
| 6 | melanoma, lesions, lesion, skin lesions, melanomas, skin cancer, pigmented, dermoscopy, malignant, benign |
| 7 | snapshot, filter, spectral imaging, imager, reconstruction, array, tunable, biomedical imaging, microscopic, sensing |
| 8 | photoacoustic, pai, photoacoustic imaging, pat, ultrasound, photoacoustic tomography, multispectral photoacoustic, acoustic, functional, imaging pai |
| 9 | severity, psoriasis, erythema, ssc, age, healthy, thickness, controls, subjects, spots |
| … | weitere 30 Cluster (siehe `best_kmeans_tfidf_2086_summary.json`) |

## Evaluation
