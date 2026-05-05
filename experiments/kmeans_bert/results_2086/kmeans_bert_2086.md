# kmeans + bert auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** Dokumente werden mit einem Bert-Model embedded (UMAP zur weiteren Dimesnionsreduktion) und per `k-means` gruppiert, um sinnvolle, gut interpretierbare Cluster (z. B. Themen oder Dokumentengruppen) zu finden. Ziel ist es, aus den Clustern verwertbare Einsichten zu gewinnen.

## Konfiguration

Die Experimentkonfiguration muss in [kmeans_bert_.yaml](../kmeans_bert.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: kmeans_bert_2086

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
| Silhouette Score | 0.6095030307769775 | |
| Davies–Bouldin Index | 0.8074986684088494 | |
| Calinski–Harabasz Index | 1043.8520036439206 | |

### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter wurden mithilfe des [Bert Interpreters](../../../src/interpretation/bert_interpreter.py) ermittelt; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung. Es wurde die Gruppierung des besten Seeds interpretiert.

Die DOI-Cluster-Zuordnung ist in der [JSON-Zusammenfassung](best_kmeans_bert_2086_summary.json) im Abschnitt `document_cluster_mapping` enthalten.

| Cluster | Top‑Wörter |
| --- | --- |
| 0 | applications automated vivo oral cancer diagnosis;deep learning, oral health diagnostics computer vision, vivo tissue classification broadband endoscopy artificial intelligence, oral cancer detection, diagnostic performance swir transillumination reflectance caries detection, detection oropharyngeal carcinoma, esophageal cancer diagnosis computer, esophageal cancer detection;objective, enamel detection quantification incipient caries, occlusal lesion detection.;.3390 |
| 1 | fabrication optical characterization gelatin- phantoms tissue, vascular phantoms reflectance, tissue phantoms medical, materials photoacoustic, phantoms photoacoustic, polyacrylamide hydrogel phantoms performance evaluation, effects phantom, tissue phantoms, fabrication phantoms, autofluorescence properties polymers |
| 2 | reconstruction algorithms, review reconstruction algorithms coded aperture snapshot, superresolution method, representative processing applications, methods segmentation, field processing, -system scene classification data fusion algorithms;battelle scientists, classification projection methods muse, learning- algorithms, tool multimodality registration applications.;.1007 |
| 3 | detector technologies, advancement photodetector technology, sensing applications, development photonics nanophotonic devices, light field manipulation mechanisms technology metasurfaces, photodetector devices, sensor applications, applications astronomy, cameras, optics |
| 4 | evaluation flap perfusion, evaluate efficacy hsi tissue perfusion assessment, impact flap perfusion, perfusion quantification, assessment tissue perfusion patients, intraoperative colon perfusion assessment, perfusion assessment, flap perfusion assessment, assessment tissue perfusion, time assessment tool tissue oxygenation micro - perfusion |
| 5 | skin imagers, skin wavelength, skin spectrum, skin processing, model- skin pigment cartography high, skin spectra, instrument skin applications, skin light sources, point performance mapping vivo skin chromophores, generation instrument skin color measurement |
| 6 | theranostic nanomedicines intrinsic fluorescence, delivery probe particles, nanoplatforms, distribution raman nanoparticles, luminescent nanoparticles, nanoplatform, set fluorescent nanomaterials, sensitivity tracking nanoparticles, nanoprobes, nanoparticle biodistribution |
| 7 | immunohistochemistry pathology, immunohistochemistry techniques, immunohistochemistry protocols, multicolor immunohistochemistry methods, multiplex analysis;tissue sections, immunohistochemistry, scoring expression immunohistochemistry, pathology platform, solutions analysis software immunohistochemistry, biomarkers histopathology specimens;aims |
| 8 | approach segmentation classification glioblastoma brain tumors, developments field -vivo brain tumour detection delineation, machine learning technique;stereotactic neuro, time classification human brain tumor, spatio- classification brain cancer detection, tool -vivo identification delineation brain tumours, deep multi - task learning framework brain tumor, tumor identification technologies, cancer segmentation mri;methods, segmentation methods brain tumor |
| 9 | assessment burn wounds, light evaluating burn wounds, depth assessment hand burns, aid assessment burn wounds, classification burn injuries, research burn severity detection method, method burn severity assessment, estimation burn depth, research application burn severity detection, h. burn severity |

## Evaluation
sehr gute Metriken, semantische Clusterevaluation steht aus