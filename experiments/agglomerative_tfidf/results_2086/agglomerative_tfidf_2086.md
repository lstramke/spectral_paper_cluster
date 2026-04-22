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
| Silhouette Score | 0.2717430889606476  |  |
| Davies–Bouldin Index | 1.6730642750088283 |  |
| Calinski–Harabasz Index | 18.07832378325674 |  |

#### Cluster-Interpretation

| Cluster | Top‑Wörter |
| --- | --- |
| 0 | prism, instrument, lambda, test, hyperspectral microscopy, lens, spectral resolution, spatial resolution, spatially, spectral imaging |
| 1 | flap, perfusion, flaps, necrosis, free flap, flap perfusion, index, sto, thi, flap necrosis |
| 2 | encryption, phase, glare, edge, fast, differential, matching, feature detection, secure, security |
| 3 | scattering, phantoms, phantom, depth, optical properties, turbid, absorption, mimicking, absorption scattering, tissue mimicking |
| 4 | noise, denoising, filter, nesma, filtering, image denoising, restoration, gaussian, prior, brain |
| 5 | deep learning, medical hyperspectral, graph, anomaly, framework, attention, 1d cnn, anomaly detection, image classification, 1d |
| 6 | cs, endoscopic imaging, light sources, endoscopic, sources, diffraction, fast, lung, source, compressive |
| 7 | foot, diabetic, healing, diabetic foot, ulcer, ulceration, ulcers, limb, dfu, diabetes |
| 8 | oxygen, saturation, oxygen saturation, sto2, hemoglobin, oxygenation, tissue oxygen, hypoxia, concentration, microcirculatory |
| 9 | optoacoustic, msot, optoacoustic tomography, multispectral optoacoustic, tomography msot, optoacoustic imaging, muscle, ultrasound, handheld, volumetric |
| … | weitere 122 Cluster (siehe `best_agglomerative_tfidf_2086_summary.json`) |

### Evaluation

