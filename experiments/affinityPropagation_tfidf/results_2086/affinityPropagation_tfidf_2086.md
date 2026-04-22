# affinity propagation + tfidf auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** Dokumente werden in TF‑IDF‑Vektoren überführt (optional LSA), anschließend wendet die Pipeline Affinity Propagation an, das automatisch repräsentative Exemplare (Cluster‑Zentren) findet. Ziel ist explorative Themenentdeckung ohne feste `k`‑Angabe; Affinity Propagation ist besonders nützlich bei kleineren Datensätzen, reagiert aber stark auf die Normalisierung der Merkmale.

## Konfiguration

Die Experimentkonfiguration muss in [affinityPropagation_tfidf.yaml](../affinityPropagation_tfidf.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:

```yaml
experiment_name: affinityPropagation_tfidf_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

affinityPropagation:
  damping_range: [0.5, 0.95]
  random_state_range: [1, 10000]
  n_trials: 120
  max_iter: 400
  convergence_iter: 15
  affinity: euclidean
  normalize: true

tfidf:
  max_features: 1000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.4
  lowercase: true
  stop_words: english
  extra_stop_words: ["hsi"]
  use_lsa: true
  lsa_components: 100

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/affinityPropagation_tfidf/results_2086
  plot_name: affinityPropagation_tfidf_2086_pca.png
  summary_name: best_affinityPropagation_tfidf_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py`
3. Clustering mit `src/clustering/affinityPropagation.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner `results_2086/` speichern

## Ergebnisse

### Plot:

![affinity propagation + tfidf PCA](affinityPropagation_tfidf_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [affinityPropagtion_tfidf_2086_pca.html](affinityPropagation_tfidf_2086_pca.html)

#### Metriken: 

Die Metriken werden in `best_affinityPropagation_tfidf_2086_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.26424601674079895 | |
| Davies–Bouldin Index | 1.7716755566247075 |  |
| Calinski–Harabasz Index | 16.440219700130623 |  |

#### Cluster-Interpretation

Die Top‑Wörter (Top‑10) pro Cluster, berechnet aus den nicht reduzierten TF‑IDF‑Features, lauten:


| Cluster | Top‑Wörter |
| ---: | --- |
| 0 | optica, publishing, group terms, optica publishing, publishing group, terms optica, optica open, publishing agreement, access publishing, open access |
| 1 | convolutional neural, convolutional, neural network, cnn, age, network cnn, using convolutional, training, body, chaotic |
| 2 | image processing, problems, gp, image data, scientific, cluster, windows, recognition, multichannel, tree |
| 3 | metasurfaces, metasurface, phase, fig, photonic, sensing, cs, amplitude, multidimensional, integral |
| 4 | lesions, pigmented, skin lesions, malignant, pigmented skin, benign, skin cancer, maflim, dermatologists, dermoscopy |
| 5 | pai, photoacoustic, photoacoustic imaging, imaging pai, dyes, oxygen, oxygen saturation, phantoms, ph, saturation |
| 6 | smartphone, mobile, rgb, device, cost, multispectral data, low cost, cameras, leverage, single shot |
| 7 | malignant, dual modality, endoscope, mouse, aotf, dual, normal, modality, tunable filter, 440 |
| 8 | microscopic, microscopic hyperspectral, attention, spatial spectral, pathological, mn, stitching, nephropathy, membranous nephropathy, membranous |
| 9 | snapshot, snapshot hyperspectral, fiber, bundle, fiber bundle, endoscope, fabricated, custom, spectrometer, double |
| … | weitere 166 Cluster (siehe `best_affinityPropagation_tfidf_2086_summary.json`) |

### Evaluation

