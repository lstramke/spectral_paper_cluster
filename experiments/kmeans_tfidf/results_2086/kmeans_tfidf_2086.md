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
  n_clusters: 10
  max_iter: 100
  tol: 0.0001
  seed_range: [1, 10000]
  n_trials: 1000

tfidf:
  max_features: 1000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.5
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
| Silhouette Score | 0.1375163197517395 | schlecht bis mäßig |
| Davies–Bouldin Index | 3.163783244649756 | zu groß |
| Calinski–Harabasz Index | 32.606495428528206 | in Ordnung |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF-Raum; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung. Es wurde die Gruppierung des besten Seeds interpretiert.

| Cluster | Top-Wörter |
| --- | --- |
| 0 | infrared, nir, near, near infrared, nm, visible, mid, detection, multispectral, ii |
| 1 | classification, learning, deep, deep learning, spectral, network, accuracy, data, model, feature |
| 2 | msot, optoacoustic, optoacoustic tomography, multispectral optoacoustic, tomography, tomography msot, multispectral, tumor, nanoparticles, contrast |
| 3 | surgery, surgical, tissue, intraoperative, tumor, hyperspectral imaging, resection, brain, time, vivo |
| 4 | cancer, tissue, breast, detection, cervical, breast cancer, images, tumor, analysis, multispectral |
| 5 | wound, healing, wound healing, tissue, diabetic, wounds, perfusion, oxygenation, hyperspectral imaging, tissue oxygenation |
| 6 | burn, depth, wounds, wound, assessment, thickness, partial, severity, tissue, hyperspectral imaging |
| 7 | image, images, data, medical, fusion, method, tongue, proposed, information, algorithm |
| 8 | skin, melanoma, lesions, multispectral, lesion, spectral, skin cancer, images, detection, diagnosis |
| 9 | perfusion, tissue, oxygenation, flap, patients, sto, tissue oxygenation, index, hyperspectral imaging, sto2 |
| 10 | optical, light, microscopy, applications, tissue, properties, multispectral, scattering, nm, high |
| 11 | photoacoustic, pai, photoacoustic imaging, pat, ultrasound, optical, tomography, vivo, resolution, tissue |
| 12 | cell, cells, immune, tumor, pd, cancer, single, expression, patients, mice |
| 13 | raman, srs, raman scattering, microscopy, scattering, spectral, cells, spectroscopy, analysis, spectra |
| 14 | patients, disease, blood, ms, treatment, liver, retinal, study, hyperspectral imaging, vascular |
| 15 | mri, brain, segmentation, magnetic, resonance, mr, magnetic resonance, images, weighted, multispectral |
| 16 | pa, pa imaging, photoacoustic, breast, ultrasound, tumor, photoacoustic imaging, melanoma, 3d, contrast |
| 17 | msi, multispectral, multispectral imaging, retinal, patients, spectral, 3d, images, data, blood |
| 18 | spectral, nm, resolution, spatial, multispectral, snapshot, applications, hyperspectral imaging, filter, range |
| 19 | fluorescence, flim, lifetime, fluorescence imaging, vivo, tissue, multispectral, excitation, microscopy, autofluorescence |

## Evaluation
