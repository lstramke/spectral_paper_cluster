# affinity propagation + bert auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** Dokumente werden mit einem Bert-Model embedded (UMAP zur weiteren Dimesnionsreduktion), anschließend wendet die Pipeline Affinity Propagation an, das automatisch repräsentative Exemplare (Cluster‑Zentren) findet. Ziel ist explorative Themenentdeckung ohne feste `k`‑Angabe; Affinity Propagation ist besonders nützlich bei kleineren Datensätzen, reagiert aber stark auf die Normalisierung der Merkmale.

## Konfiguration

Die Experimentkonfiguration muss in [affinityPropagation_bert.yaml](../affinityPropagation_bert.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:

```yaml
experiment_name: affinityPropagation_bert_2086

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

interpretation_bert:
  top_n_terms: 10
  model_name: NeuML/bioclinical-modernbert-base-embeddings
  spacy_pipeline: en_core_web_sm
  pos_pattern: "<ADJ.*>*<N.*>+"
  use_mmr: False
  diversity: 0.5
  nr_candidates: 20


outputs:
  output_dir: experiments/affinityPropagation_bert/results_2086
  plot_name: affinityPropagation_bert_2086_pca.png
  summary_name: best_affinityPropagation_bert_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

## Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/bert.py`
3. Clustering mit `src/clustering/affinityPropagation.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner `results_2086/` speichern

## Ergebnisse

### Plot:

![affinity propagation + bert PCA](affinityPropagation_bert_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [affinityPropagtion_bert_2086_pca.html](affinityPropagation_bert_2086_pca.html)

#### Metriken: 

Die Metriken werden in `best_affinityPropagation_bert_2086_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score |  0.5981753468513489 | |
| Davies–Bouldin Index | 0.8424029267128099 |  |
| Calinski–Harabasz Index | 1105.4158109244113 |  |

#### Cluster-Interpretation

Die Wörter wurden mithilfe des [Bert Interpreters](../../../src/interpretation/bert_interpreter.py) ermittelt.

| Cluster | Top‑Wörter |
| ---: | --- |
| 0 | detector technologies, light field manipulation mechanisms technology metasurfaces, sensing applications, development photonics nanophotonic devices, cameras, sensor- cameras, applications detection, optics, detector systems, metasurfaces- devices |
| 1 | diffusion model- framework transformer classification, learning architecture classification label, learning network;segmentation, learning specific general realm feature representations fusion, segmentation;semantic segmentation machine learning task, analysis highlights superiority graph convolutional networks imagery tasks, optimization transformer network super - resolution, shot classification multiscale spatial- attention, multiscale feature fusion transformer generation, feature perception framework |
| 2 | esophageal cancer diagnosis computer, ex vivo tissue classification broadband endoscopy artificial intelligence, esophageal cancer detection yolo frameworks, esophageal cancer detection;objective, esophageal cancer detection;band selection, vision early accurate cancer identification;background, endoscopic cancer, endoscopic classification spectrum, endoscopic chip, endoscopic categorization |
| 3 | microscope system biomedical applications, custom scanning system biomedical applications, camera design, cameras, micromirror device- system, optics research, technologies system, application technology, component method array periscopes, applications |
| 4 | classification model tumor tissue detection lumpectomy resection surface, cancer detection margins breast specimens, breast cancer margin assessment technique, detection cancer tissue, evaluation carcinoma margins, characterization mammary tumors noninvasive tactile sensors, analysis resection margins breast cancer, identification cancer tissue, classification cancer, margin analysis breast |
| 5 | technology applications, analysis technology, technologies, technology, micro- technology, paper reviews research status microscopic technology, use technologies, modalities applications, detection prospects, applications biomedicine |
| 6 | rapid identification infectious pathogens, detection bacteria, setup situ detection bacteria, bacteria detection technology, rapid detection common infected bacteria fluorescence effect, identification microorganisms, infections, network pathogen identification, identification agaric infection, antibiotics biofilm |
| 7 | classification diagnose tumors;deep learning, cancer;deep learning, deep learning liver cancer staging cirrhosis differentiation;liver malignancies, unsupervised spatial attention- generative adversarial network cholangiocarcinoma detection;cholangiocarcinoma, % machine learning method attention module, learning framework classification region interest pattern complex medical, convolutional neural network design brain tumor mri classification, diagnostic approach osteosarcoma bone callus deep learning;distinguishing, deep learning framework photoacoustic specimen, aware network kan robust classification |
| 8 | probe;photoacoustic tomography, photoacoustic tomography opening new paradigms biomedical, photoacoustic tomography, biomedical photoacoustics, photoacoustic, photoacoustic whole, diode- photoacoustic computed tomography, handheld photoacoustic probe label, wavelength photoacoustic system;osteoporosis, tomography systems |
| 9 | assessment liver ablation, detection analysis intestinal ischemia, biomarker assessment liver fat, evaluation liver viability hao model artificial, radiofrequency ablation liver, high precision monitoring radiofrequency ablation liver, techniques detection quantification liver, liver injury, liver viability scoring deep learning, score liver viability |
| … | weitere 113 Cluster (siehe `best_affinityPropagation_fasttext_2086_summary.json`) |

### Evaluation
Metriken sind sehr gut, semantische Clusterevaluation steht noch aus
