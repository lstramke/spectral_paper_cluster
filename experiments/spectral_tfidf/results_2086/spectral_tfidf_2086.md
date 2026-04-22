# spectral + tfidf auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** TF‑IDF‑Vektoren (optional LSA) werden über einen kNN‑Affinitätsgraph (Cosine) oder RBF‑Kernel an Spectral Clustering übergeben, um thematische Dokumentengruppen — auch nicht‑konvexe Strukturen — zu entdecken; geeignet für mittelgroße Datensätze.

## Konfiguration

Die Experimentkonfiguration muss in [spectral_tfidf.yaml](../spectral_tfidf.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: spectral_tfidf_2086

input:
  documents_path: data/raw/dataset_2086.csv
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
  lsa_components: 100

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/spectral_tfidf/results_2086
  plot_name: spectral_tfidf_2086_pca.png
  summary_name: best_spectral_tfidf_2086_summary.json
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
5. Outputs: Plot und Summary im Unterordner unter `results_2086/` speichern

## Ergebnisse

### Plot:

![spectral + tfidf PCA](spectral_tfidf_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [spectral_tfidf_2086_pca.html](spectral_tfidf_2086_pca.html)

### Metriken:

Die Metriken werden in `best_spectral_tfidf_2086_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.11032392084598541 |  |
| Davies–Bouldin Index | 2.603746602659673 |  |
| Calinski–Harabasz Index | 28.54055405437507 | |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster (Top‑10), berechnet aus den nicht reduzierten TF‑IDF‑Features:

| Cluster | Top‑Wörter |
| ---: | --- |
| 0 | fluorescence, flim, lifetime, fluorescence imaging, oral, autofluorescence, multispectral, vivo, tissue, detection |
| 1 | tongue, medicine, color, diagnosis, images, information, segmentation, method, traditional, spectral |
| 2 | brain, mri, segmentation, magnetic, ms, mr, magnetic resonance, resonance, images, multispectral |
| 3 | wound, healing, wound healing, tissue, diabetic, wounds, oxygenation, perfusion, hyperspectral imaging, tissue oxygenation |
| 4 | spectral, image, multispectral, tissue, optical, images, hyperspectral imaging, data, high, applications |
| 5 | raman, srs, raman scattering, scattering, microscopy, cells, spectral, chemical, spectroscopy, free |
| 6 | liver, thermal, index, tissue, infrared, spectral, hyperspectral imaging, treatment, kidney, assessment |
| 7 | retinal, fundus, spectral, disease, camera, images, nm, multimodal, oxygen, saturation |
| 8 | pa, pa imaging, photoacoustic, ultrasound, tumor, contrast, photoacoustic imaging, molecular, optical, breast |
| 9 | burn, depth, wounds, wound, thickness, assessment, partial, severity, hyperspectral imaging, msi |
| 10 | melanoma, lesions, lesion, skin, diagnostic, multispectral, specificity, detection, diagnosis, sensitivity |
| 11 | fusion, image fusion, image, images, transform, proposed, multi, multispectral, multispectral image, coefficients |
| 12 | perfusion, flap, tissue, oxygenation, sto, index, patients, tissue oxygenation, hyperspectral imaging, surgery |
| 13 | classification, learning, data, spectral, deep, model, network, image, deep learning, images |
| 14 | skin, lesions, severity, spectral, assessment, multispectral, non, hyperspectral imaging, images, skin cancer |
| 15 | photoacoustic, pai, photoacoustic imaging, ultrasound, optical, vivo, contrast, tissue, multispectral, absorption |
| 16 | optoacoustic, msot, optoacoustic tomography, multispectral optoacoustic, tomography, tomography msot, multispectral, contrast, tissue, disease |
| 17 | tensor, low, data, dimensionality, matrix, medical hyperspectral, reduction, medical, images, framework |
| 18 | cell, cells, immune, pd, tumor, cancer, patients, single, expression, cellular |
| 19 | breast, breast cancer, cancer, tissue, detection, tumor, analysis, margin, patients, images |

## Evaluation
