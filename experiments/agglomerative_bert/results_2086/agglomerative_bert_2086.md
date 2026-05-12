# agglomerative + bert auf 2086

## Kurzüberblick

- **Kurzbeschreibung:**  Dokumente werden mit einem Bert-Model embedded (UMAP zur weiteren Dimesnionsreduktion), um Dokumente über kosinus‑basierte Ähnlichkeit zu gruppieren. Die agglomerative Clusterung schneidet den Dendrogramm‑Baum über einen Distanz‑Threshold, sodass thematisch ähnliche Dokumentgruppen extrahiert werden können. Ziel ist die explorative Identifikation von Themen und die anschließende Interpretierbarkeit der Cluster.

## Konfiguration

Die Experimentkonfiguration muss in [agglomerative_bert.yaml](../agglomerative_bert.yaml) eingetragen sein.

Die Konfiguration für das hier dargestellte Ergebnis ist:
```yaml
experiment_name: agglomerative_bert_2086

input:
  documents_path: data/raw/dataset_2086_withDOI.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

agglomerative:
  distance_threshold_range: [0.000001, 0.001]
  n_trials: 1000
  metric: cosine
  linkage: average
  compute_full_tree: true

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
  output_dir: experiments/agglomerative_bert/results_2086
  plot_name: agglomerative_bert_2086_pca.png
  summary_name: best_agglomerative_bert_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/bert.py`
3. Clustering mit `src/clustering/agglomerativeClustering.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner `results_2086/` speichern

### Ergebnisse

#### Plot:

![agglomerative + bert PCA](agglomerative_bert_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [agglomerative_bert_2086_pca.html](agglomerative_bert_2086_pca.html)

#### Metriken:

Die Metriken werden in `best_agglomerative_bert_2086_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score |  0.5675085186958313  |  |
| Davies–Bouldin Index | 0.7875113003138225 |  |
| Calinski–Harabasz Index | 922.7672424203865 |  |

#### Cluster-Interpretation

Die Wörter wurden mithilfe des [Bert Interpreters](../../../src/interpretation/bert_interpreter.py) ermittelt.

| Cluster | Top‑Wörter |
| --- | --- |
| 0 | optical modalities; techniques applications animal man;optical brain; optical; optical brain vivo; technologies; technology; optical instrumentation engineers; techniques; applications |
| 1 | solutions machine vision; detection algorithms; detection field; -system scene classification data fusion algorithms;battelle scientists; spatial lasso applications unmixing biomedical; filtering technique model pixel; art super - resolution techniques; technology analysis tasks; reconstruction algorithms |
| 2 | cameras; camera design; development snapshot imager microlens array; snapshot cameras; optics research; component method array periscopes; snapshot techniques; custom scanning system biomedical applications; bioimaging applications; time enhancement methods snapshot cameras |
| 3 | multivariate chemometrics analysis; chemometrics analysis; chemometrics analyses; phasor analysis; denoising impact classification techniques; collection thousands absorption signals handful; spectra identification; analysis methods; tool deconvoluting chemical effects; models quantification cell parameters |
| 4 | micro - raman spectroscopy; stimulated raman scattering microscopy; light sheet raman micro - spectroscopy; applications chemical resolution visualization; development microscopy spectroscopy techniques; cell raman spectroscopy; contrast raman spectroscopy; scanning techniques raman; scale raman micro -; micro - spectroscopies |
| 5 | systems measurement perfusion oxygenation; technique blood oxygenation research; calibration validation scheme vivo spectroscopic tissue oxygenation; noninvasive assessment retinal vascular oxygen content; technique measurement blood oxygen saturation vivo; msi oximetry vivo applications; vascular oxygen measurements; measurement tissue oxygenation; practice measurement tissue oxygenation; mapping oxygen saturation tissue |
| 6 | camera application; sensor; cameras; skin spectra; detection; sensors; camera; applications; camera target; evaluation framework sfa cameras |
| 7 | probe;photoacoustic tomography; absorption vivo;photoacoustic tomography; photoacoustic tomography opening new paradigms biomedical; photoacoustic tomography; biomedical photoacoustics; photoacoustic whole; photoacoustic; diode- photoacoustic computed tomography; wavelength photoacoustic system; potential photoacoustic radiation |
| 8 | evaluation laparoscopic system;background; laparoscopic system; guidance;intraoperative fluorescence; laparoscopic hsi system; tissue laparoscopic system; perspective surgery; new intraoperative tools; parameter laparoscopic system; photoacoustic irregular hepatectomy navigation; optical- instrument |
| 9 | tissue segmentation liver head neck surgeries machine learning;aim; machine learning models; learning 3d tumor modeling; augmented reality computational surgical; tissue segmentation surgery; machine learning model; arthroscopic scene segmentation; machine learning; machine learning tools; machine learning algorithms |
| … | weitere 68 Cluster (siehe `best_agglomerative_bert_2086_summary.json`) |

### Evaluation

Metriken sind sehr gut, semantische Clusterevaluation steht aus