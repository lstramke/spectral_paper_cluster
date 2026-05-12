# spectral + bert auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** Dokumente werden mit einem Bert-Model embedded (UMAP zur weiteren Dimesnionsreduktion) und über einen kNN‑Affinitätsgraph (Cosine) oder RBF‑Kernel an Spectral Clustering übergeben, um thematische Dokumentengruppen — auch nicht‑konvexe Strukturen — zu entdecken; geeignet für mittelgroße Datensätze.

## Konfiguration

Die Experimentkonfiguration muss in [spectral_bert.yaml](../spectral_bert.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: spectral_bert_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

spectral:
  n_clusters_range: [5, 40]
  affinity: nearest_neighbors
  eigen_solver: arpack
  assign_labels: kmeans
  n_init: 10
  gamma: 1.0
  n_neighbors_range: [5, 20]
  random_state_range: [0, 10000]
  n_jobs: 1
  n_trials: 400

bert:
  model_name: NeuML/bioclinical-modernbert-base-embeddings
  device: cpu
  batch_size: 8
  normalize: True
  show_progress: False
  umap_n_components: 100
  umap_random_state: 42
  preprocess_with_tfidf: true
  tfidf_max_df: 0.4
  tfidf_max_features: 5000
  spacy_pipeline: en_core_web_sm

interpretation_bert:
  top_n_terms: 10
  model_name: NeuML/bioclinical-modernbert-base-embeddings
  spacy_pipeline: en_core_web_sm
  pos_pattern: "<ADJ.*>*<N.*>+"
  use_mmr: False
  diversity: 0.5
  nr_candidates: 20

outputs:
  output_dir: experiments/spectral_bert/results_2086
  plot_name: spectral_bert_2086_pca.png
  summary_name: best_spectral_bert_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/bert.py`
3. Clustering mit `src/clustering/spectralClustering.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner unter `results_2086/` speichern

## Ergebnisse

### Plot:

![spectral + bert PCA](spectral_bert_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [spectral_bert_2086_pca.html](spectral_bert_2086_pca.html)

### Metriken:

Die Metriken werden in `best_spectral_bert_2086_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.5480008125305176 |  |
| Davies–Bouldin Index | 0.8051023253876367 |  |
| Calinski–Harabasz Index | 972.8247780991526 | |

### Cluster-Interpretation

Die Wörter wurden mithilfe des [Bert Interpreters](../../../src/interpretation/bert_interpreter.py) ermittelt.

| Cluster | Top‑Wörter |
| ---: | --- |
| 0 | range applications crop plant sciences, characterisation crops plants, application precision agriculture, remote sensing monitoring crop disease, crop plant sciences, classification medicinal plant, analysis precision agriculture, hyperspectra used recognize black goji berry nitraria, assessment crop, discrimination vegetation areas |
| 1 | photonics devices, detector technologies, light field manipulation mechanisms technology metasurfaces, sensing applications, applications astronomy, cameras, development photonics nanophotonic devices, sensing platforms, applications detection, sensor- cameras |
| 2 | fluorescence microscopy biological, microscope system biomedical applications, spr fluorescence detection techniques, fluorescence applications, fluorescence microscopy, fluorescent microscopy, mode microscopic imager sensing biological samples, animal microscopy, microscopy biomedical, fluorescence microscopes |
| 3 | skin detection, skin assessment, method skin assessment, skin assessment tool abilities, skin diagnostics;significance, skin diagnostics, time monitoring skin features, skin imagers, analysis skin, analysis skin lesions polarization |
| 4 | classification diagnose tumors;deep learning, cancer;deep learning, deep learning liver cancer staging cirrhosis differentiation;liver malignancies, unsupervised spatial attention- generative adversarial network cholangiocarcinoma detection;cholangiocarcinoma, learning framework classification region interest pattern complex medical, % machine learning method attention module, diagnostic approach osteosarcoma bone callus deep learning;distinguishing, convolutional neural network design brain tumor mri classification, deep learning framework photoacoustic specimen |
| 5 | tumor identification technologies, tissue segmentation liver head neck surgeries machine learning;aim, cancer segmentation mri;methods, learning 3d tumor modeling, time classification human brain tumor, hsi analysis machine learning, developments field -vivo brain tumour detection delineation, intraoperative tool -vivo identification delineation brain tumours, spatio- classification brain cancer detection, approach segmentation classification glioblastoma brain tumors |
| 6 | tomography absorption, fluorescence molecular tomography, view fluorescence molecular tomography nir, tomography;bioluminescence tomography, tomography platform, tomography system, applications microscopy, single pixel bioluminescence tomography compressive sensing;photonics, applications microscopy, coherence tomography |
| 7 | learning denoising methods, art denoising methods, data;kernel- dimensionality reduction, denoising framework, network denoising, replacement denoising framework, denoising model, denoising approach, tensor data sound approach completing, denoising methods |
| 8 | applications automated vivo oral cancer diagnosis;deep learning, use technology tongue diagnosis, oral health diagnostics computer vision, sensor tongue diagnosis;purpose, information tongue diagnosis, sensor system tcm tongue diagnosis, tongue tumor detection medical, oral cancer detection, tongue coating grading identification deep learning data;tongue diagnosis, analysis tongue |
| 9 | modalities applications, technology applications, application instrumentation, bioimaging applications, spectroscopic applications, applications, applications targets, applications;significance, camera application, technology assessment |
| … | weitere 26 Cluster (siehe `best_spectral_bert_2086_summary.json`) |

## Evaluation
Metriken sind sehr gut. semantische Clusterevaluation steht noch aus