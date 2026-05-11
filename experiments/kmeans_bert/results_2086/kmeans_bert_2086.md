# kmeans + bert auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** Dokumente werden mit einem Bert-Model embedded (UMAP zur weiteren Dimesnionsreduktion) und per `k-means` gruppiert, um sinnvolle, gut interpretierbare Cluster (z. B. Themen oder Dokumentengruppen) zu finden. Ziel ist es, aus den Clustern verwertbare Einsichten zu gewinnen.

## Konfiguration

Die Experimentkonfiguration muss in [kmeans_bert_.yaml](../kmeans_bert.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: kmeans_bert_2086

input:
  documents_path: data/raw/dataset_2086_withDOI.csv
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
  output_dir: experiments/kmeans_bert/results_2086
  plot_name: kmeans_bert_2086_pca.png
  summary_name: best_kmeans_bert_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/bert.py`
3. `k-means` Clustering (siehe `src/clustering/kmeans.py`)
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: PCA wird zur 3D-Visualisierung nach dem Clustering angewendet. Plot und Metrik-JSON werden zusammen in einem Unterordner `results_2086/` abgelegt.

## Ergebnisse

Das Ergebnisbild und die zugehörige JSON-Zusammenfassung werden im Experiment-Unterordner unter `results_2086/` abgelegt.

### Plot (PCA):

![kmeans + bert PCA](kmeans_bert_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [kmeans_bert_2086_pca.html](kmeans_bert_2086_pca.html)

### Metriken:

Die Metriken für alle Zufallswerte werden in [`kmeans_bert_2086_all_runs.json`](kmeans_bert_2086_all_runs.json) gespeichert. Die Details zum besten Lauf stehen zusätzlich in [`best_kmeans_bert_2086_summary.json`](best_kmeans_bert_2086_summary.json). Für den aktuellen besten Lauf ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.6144372224807739 | |
| Davies–Bouldin Index | 0.8212178842701547 | |
| Calinski–Harabasz Index | 1047.2752308703832 | |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter wurden mithilfe des [Bert Interpreters](../../../src/interpretation/bert_interpreter.py) ermittelt; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung. Es wurde die Gruppierung des besten Seeds interpretiert.

Die DOI-Cluster-Zuordnung ist in der [JSON-Zusammenfassung](best_kmeans_bert_2086_summary.json) im Abschnitt `document_cluster_mapping` enthalten.

| Cluster | Top‑Wörter |
| --- | --- |
| 0 | classification diagnose tumors;deep learning, cancer;deep learning, unsupervised spatial attention- generative adversarial network cholangiocarcinoma detection;cholangiocarcinoma, convolutional neural network design brain tumor mri classification, learning framework classification region interest pattern complex medical, % machine learning method attention module, deep learning framework photoacoustic specimen, deep learning, machine learning algorithms, aware network kan robust classification |
| 1 | information field tissue characterization, approach analysis, techniques methodologies, applications method, applications, spectra tissue types, analysis color, spatial lasso applications unmixing biomedical, field tissue characterization, technologies |
| 2 | handheld optoacoustic tomography;background, optoacoustic tomography enables, application optoacoustic tomography, optoacoustic tomography functional vascular research;microcirculatory impairment, applicability optoacoustic tomography, optoacoustic tomography, optoacoustic tomography functional assessment gastrointestinal, optoacoustic tomography muscle perfusion oxygenation, performance optoacoustic tomography, technology ultrasound tomography |
| 3 | assessment burn wounds, light evaluating burn wounds, depth assessment hand burns, aid assessment burn wounds, classification burn injuries, research burn severity detection method, method burn severity assessment, estimation burn depth, research application burn severity detection, h. burn severity |
| 4 | micro - raman spectroscopy, applications raman microscopy, stimulated raman scattering microscopy;significance field, use raman microscopy life sciences, light sheet raman micro - spectroscopy, cell raman spectroscopy, applications chemical resolution visualization;raman spectroscopy, development microscopy spectroscopy techniques, contrast raman spectroscopy, scale raman micro - |
| 5 | learning denoising methods, art denoising methods, adaptive denoising, network denoising, replacement denoising framework, hsi denoising methods, denoising method, denoising methods, restoration denoising techniques, denoising framework |
| 6 | fluorescence microscopy biological, fluorescent signals fluorescence microscopy studies, fluorescence microscopy, fluorescence applications, fluorescent microscopy, microscope system biomedical applications, fluorescence microscopes, speed fluorescence lifetime implementation vivo applications;fluorescence lifetime microscopy, fluorescence organoscopes, animal microscopy |
| 7 | evaluation flap perfusion, evaluate efficacy hsi tissue perfusion assessment, perfusion quantification, impact flap perfusion, assessment tissue perfusion patients, perfusion assessment, flap perfusion assessment, intraoperative colon perfusion assessment, assessment tissue perfusion, skin perfusion measurement |
| 8 | applications cancer detection, classification model tumor tissue detection lumpectomy resection surface, cancer detection margins breast specimens, evaluation carcinoma margins.;.1007, detection cancer tissue, breast cancer margin assessment technique, system applications cancer detection, characterization mammary tumors noninvasive tactile sensors, analysis resection margins breast cancer, identification cancer tissue |
| 9 | detector technologies, sensing applications, advancement photodetector technology, development photonics nanophotonic devices, photodetector devices, light field manipulation mechanisms technology metasurfaces, sensor applications, applications astronomy, silicon- sensors, applications detection |

## Evaluation
sehr gute Metriken, semantische Clusterevaluation steht aus