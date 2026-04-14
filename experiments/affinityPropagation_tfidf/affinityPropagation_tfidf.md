# affinity propagation + tfidf

## Kurzüberblick

- **Kurzbeschreibung:** Dokumente werden in TF‑IDF‑Vektoren überführt (optional LSA), anschließend wendet die Pipeline Affinity Propagation an, das automatisch repräsentative Exemplare (Cluster‑Zentren) findet. Ziel ist explorative Themenentdeckung ohne feste `k`‑Angabe; Affinity Propagation ist besonders nützlich bei kleineren Datensätzen, reagiert aber stark auf die `preference`‑Einstellung und auf Normalisierung der Merkmale.

## Konfiguration

Die Experimentkonfiguration liegt in [affinityPropagation_tfidf.yaml](affinityPropagation_tfidf.yaml).

```yaml
experiment_name: affinityPropagation_tfidf

input:
  documents_path: data/raw/data_db_raw.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

affinityPropagation:
  damping: 0.5
  max_iter: 200
  convergence_iter: 15
  affinity: euclidean
  random_state: 42
  normalize: true

tfidf:
  max_features: 1000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.5
  lowercase: true
  stop_words: english
  extra_stop_words: ["hsi"]
  use_lsa: true
  lsa_components: 100

interpretation:
  top_n_terms: 10

outputs:
  output_dir: experiments/affinityPropagation_tfidf/outputs
  plot_name: affinityPropagation_tfidf_pca.png
  summary_name: best_affinityPropagation_tfidf_summary.json
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
5. Outputs: Plot und Summary im Unterordner `outputs/` speichern

## Ergebnisse

### Plot:

![affinity propagation + tfidf PCA](outputs/affinityPropagation_tfidf_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [outputs/affinityPropagtion_tfidf_pca.html](outputs/affinityPropagation_tfidf_pca.html)

#### Metriken: 

Die in der JSON gespeicherten Kennzahlen direkt auswerten.

Die Metriken werden in `outputs/best_affinityPropagation_tfidf_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.0943 | schwach getrennte Cluster |
| Davies–Bouldin Index | 2.0359 | mittlere bis deutliche Überlappung  |
| Calinski–Harabasz Index | 2.0412 | niedriger bis schwache Clusterstruktur |

#### Cluster-Interpretation

Die Top‑Wörter (Top‑10) pro Cluster, berechnet aus den nicht reduzierten TF‑IDF‑Features, lauten:

| Cluster | Top‑Wörter |
| ---: | --- |
| 0 | biological, high, proposed, tissue, resolution, images, tissues, using, use, different |
| 1 | patients, studies, vivo, systems, multispectral, reported, small, surgical, systematic, current |
| 2 | disease, disorders, diseases, field, data, monitoring, overview, advances, current, clinical |
| 3 | medical, medical applications, research, challenges, clinical, limitations, field, future, study, technology |
| 4 | perfusion, clinical, surgery, measurements, literature, gastrointestinal, promising, results, spectral imaging, tissue |
| 5 | brain, tissue, technology, information, biological, diagnosis, recent, acquisition, disease diagnosis, disease |
| 6 | cancer, studies, accuracy, detection, meta, sensitivity, meta analysis, techniques, computer aided, aided |
| 7 | skin, cancer, skin cancer, color, light, compared, detection, systematic, multispectral, systematic review |
| 8 | multispectral, vision, technology, spectroscopy, capabilities, technologies, different, based, lesions, multispectral imaging |
| 9 | learning, medical, images, algorithms, techniques, image, data, various, machine, systems |

### Evaluation

Die Kennzahlen zeigen eine schwache Clusterstruktur: Silhouette = 0.0943 (zweitbester Wert), Davies–Bouldin ≈ 2.036, Calinski–Harabasz ≈ 2.041. Affinity Propagation erzeugte viele kleine Cluster.

Kurzfristig testen: `preference` systematisch variieren (Median/Quantile) dafür mit aufnehmen in dei config,
