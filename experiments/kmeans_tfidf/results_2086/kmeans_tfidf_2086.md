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
| Silhouette Score | 0.20143121480941772 | bis mäßig |
| Davies–Bouldin Index | 2.620221613949932 | groß |
| Calinski–Harabasz Index | 25.6402152886520 | in Ordnung |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF-Raum; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung. Es wurde die Gruppierung des besten Seeds interpretiert.

| Cluster | Top-Wörter |
| --- | --- |
| 0 | nir, nir ii, ii, vis, infrared nir, liver, nir imaging, photodetectors, vis nir, kidney |
| 1 | pathology, prostate, staining, mif, immunohistochemistry, prostate cancer, ihc, expression, digital, automated |
| 2 | melanoma, lesions, lesion, melanomas, skin lesions, pigmented, skin cancer, dermoscopy, malignant, benign |
| 3 | polarization, metasurface, metasurfaces, sensing, filter, thz, terahertz, biomedical imaging, compressive, sensor |
| 4 | segmentation, attention, net, microscopic, framework, deep learning, microscopic hyperspectral, transformer, hyperspectral image, module |
| 5 | photoacoustic, pat, photoacoustic tomography, ultrasound, photoacoustic imaging, multispectral photoacoustic, acoustic, functional, absorption, tomography pat |
| 6 | source, endoscopy, oct, light source, illumination, endoscopic, calibration, caries, 3d, sources |
| 7 | perfusion, flap, oxygenation, index, sto, tissue oxygenation, nir, twi, flaps, oxygen |
| 8 | mri, brain, segmentation, mr, magnetic, resonance, magnetic resonance, weighted, ms, matter |
| 9 | optoacoustic, msot, optoacoustic tomography, multispectral optoacoustic, tomography msot, optoacoustic imaging, handheld, functional, volumetric, healthy |
| 10 | ct, ray, material, detector, energy, photon counting, photon, counting, cdte, edge |
| 11 | burn, burns, burn depth, depth, wounds, burn wounds, wound, thickness, partial thickness, severity |
| 12 | cnn, band, convolutional, deep learning, autofluorescence, convolutional neural, neural network, oral, band selection, selection |
| 13 | fusion, image fusion, fused, transform, multispectral image, fusion network, fused image, coefficients, end, neural network |
| 14 | ai, artificial, intelligence, artificial intelligence, intelligence ai, ai based, healthcare, esophageal, mc, networks |
| 15 | content, machine learning, origin, leaf, plant, prediction, stem, support, vegetation, forest |
| 16 | msi, imaging msi, 3d msi, retinal, biopsy, 3d, mass spectrometry, mass, spectrometry, comb |
| 17 | attribution, creative commons, commons attribution, commons, creative, license, spie creative, published spie, published, authors published |
| 18 | registration, unmixing, nerve, image processing, image registration, veins, multichannel, algorithms, linear, parallel |
| 19 | pa, pa imaging, photoacoustic, photoacoustic pa, ultrasound, multispectral pa, photoacoustic imaging, artery, fluence, breast |
| … | weitere 20 Cluster (siehe `best_kmeans_tfidf_2086_summary.json`) |

## Evaluation
