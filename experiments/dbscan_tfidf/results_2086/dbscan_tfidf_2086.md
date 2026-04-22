# dbscan + tfidf auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** TF‑IDF-Feature-Extraktion gefolgt von DBSCAN-Clustering. Ziel ist, dichte, thematische Gruppen ohne feste Clusteranzahl zu identifizieren.

## Konfiguration

Die Experimentkonfiguration muss in [dbscan_tfidf.yaml](../dbscan_tfidf.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: dbscan_tfidf_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

dbscan:
  eps_range: [0.3, 1.0]
  min_samples_range: [2, 20]
  n_trials: 400
  metric: cosine
  leaf_size: 30
  p:
  n_jobs:

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
  output_dir: experiments/dbscan_tfidf/results_2086
  plot_name: dbscan_tfidf_2086_pca.png
  summary_name: best_dbscan_tfidf_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7

```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py`
3. Clustering mit `src/clustering/dbscan.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: PNG-Plot und Summary-JSON im Unterordner `results_2086/` speichern

## Ergebnisse

### Plot:

![dbscan + tfidf PCA](dbscan_tfidf_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [dbscan_tfidf_pca.html](dbscan_tfidf_2086_pca.html)

### Metriken:

Die Metriken werden in `best_dbscan_tfidf_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.06773672997951508 | |
| Davies–Bouldin Index | 2.1881803095409063 | |
| Calinski–Harabasz Index | 21.951147621359112 | |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF-Raum; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung.

| Cluster | Top-Wörter |
| --- | --- |
| -1 | deep learning, 3d, msi, nir, reconstruction, noise, band, sensing, rgb, wavelengths |
| 0 | raman, srs, raman scattering, scattering, stimulated raman, stimulated, raman imaging, chemical, spectroscopy, label |
| 1 | burn, burn depth, burns, depth, wounds, burn wounds, wound, thickness, partial thickness, severity |
| 2 | segmentation, brain, mri, magnetic, mr, magnetic resonance, resonance, image segmentation, weighted, microscopic |
| 3 | fusion, image fusion, fused, transform, spatial spectral, fused image, coefficients, multispectral image, realm, wavelet |
| 4 | pai, photoacoustic, photoacoustic imaging, multispectral photoacoustic, ultrasound, imaging pai, acoustic, phantoms, oxygen, dyes |
| 5 | tongue, tongue diagnosis, medicine, coating, tongue images, tongue color, kampo, tcm, segmentation, chinese medicine |
| 6 | tensor, rank, low rank, dimensionality, dimensionality reduction, embedding, reduction, matrix, decomposition, completion |
| 7 | save, wli, nbi, esophageal, esophageal cancer, ec, dysplasia, f1, recall, yolov5 |
| 8 | cervical, cervical cancer, colposcopy, cin, cervix, neoplasia, acetic, acetic acid, intraepithelial, screening |
| 9 | flim, lifetime, fluorescence lifetime, lifetime imaging, oral, multispectral fluorescence, biochemical, multispectral flim, autofluorescence, imaging flim |
| 10 | msot, optoacoustic, optoacoustic tomography, multispectral optoacoustic, nanoparticles, tomography msot, optoacoustic imaging, gold, muscle, gold nanoparticles |
| 11 | perfusion, wound, flap, oxygenation, healing, sto, tissue oxygenation, wound healing, index, oxygen |
| 12 | pa, pa imaging, photoacoustic, photoacoustic pa, ultrasound, multispectral pa, breast, artery, motion, cp |
| 13 | breast, breast cancer, margin, resection, margin assessment, bc, breast tissue, specimens, malignant, normal |
| 14 | ct, ray, material, detector, photon counting, photon, counting, cdte, energy, detectors |
| 15 | melanoma, lesions, melanomas, lesion, pigmented, dermoscopy, nevi, specificity, benign, malignant |
| 16 | thermal, thermal imaging, temperature, thermography, indicators, clinical improvement, deoxyhemoglobin, limbs, body, imagery |
| 17 | immune, pd, cd8, pd l1, l1, immunohistochemistry, mif, expression, multiplex, microenvironment |
| 18 | pat, photoacoustic tomography, photoacoustic, tomography pat, acoustic, image reconstruction, reconstruction, functional, organ, functional molecular |

## Evaluation

1392 als noise markiert