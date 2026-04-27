# gaussianMixture + fasttext auf 2086

## Kurzüberblick

- **Kurzbeschreibung:** kurze, natürliche Beschreibung des Experiments und der Zielsetzung.

- **Kurzbeschreibung:** Unüberwachtes Clustering von Dokumenten mithilfe von TF‑IDF (mit optionaler LSA) gefolgt von einem Gaussian Mixture Model (GMM). Ziel ist es, semantische Gruppen im Korpus zu identifizieren und die Clusterqualität mit etablierten Metriken zu bewerten.

## Konfiguration

Die Experimentkonfiguration liegt in [gaussianMixture_fasttext.yaml](../gaussianMixture_fasttext.yaml).

```yaml
experiment_name: gaussianMixture_fasttext_2086

input:
  documents_path: data/raw/dataset_2086.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

gaussianMixture:
  n_components_range: [5, 20]
  tol: 0.001
  reg_covar: 1e-6
  max_iter: 200
  n_init: 10
  init_params: kmeans
  random_state_range: [1, 10000]
  covariance_type: full
  n_trials: 400

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/gaussianMixture_fasttext/results_2086
  plot_name: gaussianMixture_fasttext_2086_pca.png
  summary_name: best_gaussianMixture_fasttext_2086_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/fasttext.py`
3. Clustering mit `src/clustering/gaussianMixture.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: Plot und Summary im Unterordner `results_2086/` speichern

## Ergebnisse

### Plot:

![GaussianMixture + fasttext PCA](gaussianMixture_fasttext_2086_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [gaussianMixture_fasttext_2086_pca.html](gaussianMixture_fasttext_2086_pca.html)

### Metriken: 

Die Metriken werden in `best_gaussianMixture_fasttext_2086_summary.json` gespeichert. Für das aktuelle Experiment ergeben sich folgende Werte:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.1393960565328598 | |
| Davies–Bouldin Index | 2.9197658266087227 | |
| Calinski–Harabasz Index | 151.957795087624 | |

### Cluster-Interpretation
Für die Interpretation wurden die Top‑Wörter aus dem nicht reduzierten TF‑IDF‑Raum verwendet; die zugehörigen Gewichte finden sich in `best_gaussianMixture_fasttext_2086_summary.json`.

| Cluster | Top‑Wörter |
| --- | --- |
| 0 | imaging, based, method, medical, information, learning, methods, classification, proposed, spatial |
| 1 | imaging, tissue, tumor, cell, vivo, cancer, cells, fluorescence, clinical, optical |
| 2 | imaging, nm, optical, resolution, high, light, fluorescence, wavelength, raman, based |
| 3 | patients, imaging, tissue, study, perfusion, clinical, skin, nm, lesions, sensitivity |
| 4 | tissue, imaging, classification, skin, patients, tumor, cancer, burn, clinical, tongue |

## Evaluation
In 3 Clustern ist das top wort Imaging, Metriken auch nicht gut