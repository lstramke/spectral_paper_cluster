# Experiments

## Überblick

Zusammenfassung und Auswertung der durchgeführten Clustering-Experimente.

### Metriken

Zur Bewertung der Clusterqualität werden drei etablierte Kennzahlen verwendet. Sie liefern Hinweise darauf, ob die resultierenden Gruppen intern kompakt, untereinander getrennt und damit statistisch wie inhaltlich plausibel sind.
Die Kennzahlen werden im Experiment aus den Clusterlabels und Feature-Vektoren mit Funktionen aus `sklearn.metrics` berechnet.

#### **Silhouette Score**

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Dabei ist $a(i)$ der durchschnittliche Abstand eines Punkts zu den anderen Punkten im eigenen Cluster und $b(i)$ der kleinste durchschnittliche Abstand zu einem anderen Cluster. Der Wert liegt zwischen $-1$ und $1$. In der Praxis gelten Werte über etwa $0.5$ oft als brauchbar bis gut, Werte nahe $0$ als Hinweis auf überlappende Cluster und negative Werte als deutliches Warnsignal für Fehlzuordnungen. Werte nahe $1$ sprechen für eine sehr saubere Trennung.

#### **Davies–Bouldin Index**

$$
DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \frac{S_i + S_j}{M_{ij}}
$$

Hier beschreibt $S_i$ die Streuung innerhalb von Cluster $i$ und $M_{ij}$ den Abstand zwischen den Zentren der Cluster $i$ und $j$. Der Index ist nach unten besser: kleine Werte bedeuten kompakte Cluster mit gutem Abstand zueinander. In vielen Anwendungen sind Werte unter etwa $1$ bereits recht ordentlich, während Werte deutlich über $2$ oder $3$ häufig auf überlappende oder instabile Cluster hindeuten.

#### **Calinski–Harabasz Index**

$$
CH = \frac{\mathrm{Tr}(B_k)/(k-1)}{\mathrm{Tr}(W_k)/(n-k)}
$$

Dabei steht $B_k$ für die Streuung zwischen den Clustern und $W_k$ für die Streuung innerhalb der Cluster. Höhere Werte sind besser, weil sie bedeuten, dass die Cluster intern kompakt und untereinander gut getrennt sind. Absolutwerte sind allerdings stark von Datenmenge, Dimensionalität und Clusteranzahl abhängig; wichtig ist daher vor allem der relative Vergleich zwischen verschiedenen Einstellungen. Größere Werte sind in der Regel besser, sehr kleine Werte deuten auf eine schwache Clusterstruktur hin.

Zusammenfassend gilt: Für eine gute Clusterstruktur sollten Silhouette und Calinski–Harabasz eher hoch sein, während der Davies–Bouldin Index eher niedrig sein sollte. Praktisch spricht das für kompakte, gut getrennte Cluster; niedrige oder unklare Werte deuten dagegen auf überlappende oder schwach ausgeprägte Gruppen hin. Gemeinsam liefern die Kennzahlen so eine verständliche Einschätzung der Clusterqualität.


## kmeans + tfidf

### Kurzüberblick

- **Kurzbeschreibung:** Texte werden in TF‑IDF‑Vektoren umgewandeln und per `k-means` gruppiert, um sinnvolle, gut interpretierbare Cluster (z. B. Themen oder Dokumentengruppen) zu finden. Ziel ist es, aus den Clustern verwertbare Einsichten zu gewinnen.

### Konfiguration

Die Experimentkonfiguration liegt in [configs/kmeans_tfidf.yaml](configs/kmeans_tfidf.yaml).

```yaml
experiment_name: kmeans_tfidf

input:
  documents_path: data/raw/data_db_raw.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

kmeans:
  n_clusters: 10
  max_iter: 100
  tol: 0.0001
  seed_range: [1, 100]

tfidf:
  max_features: 1000
  ngram_range: [1, 2]
  min_df: 5
  max_df: 0.5
  lowercase: true
  stop_words: english
  extra_stop_words: ["don", "like"]
  use_lsa: true
  lsa_components: 100

interpretation:
  top_n_terms: 10

outputs:
  output_dir: outputs/kmeans_tfidf
  plot_name: kmeans_tfidf_pca.png
  summary_name: best_kmeans_tfidf_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py` (TF‑IDF, optional LSA)
3. `k-means` Clustering (siehe `src/clustering/kmeans.py`)
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: PCA wird zur 3D-Visualisierung nach dem Clustering angewendet. Plot und Metrik-JSON werden zusammen in einem Unterordner unter `outputs/` abgelegt.

### Ergebnisse

Das Ergebnisbild und die zugehörige JSON-Zusammenfassung werden im Experiment-Unterordner unter `outputs/kmeans_tfidf/` abgelegt.

#### Plot (PCA):

![kmeans + tfidf PCA](outputs/kmeans_tfidf/kmeans_tfidf_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [outputs/kmeans_tfidf/kmeans_tfidf_pca.html](outputs/kmeans_tfidf/kmeans_tfidf_pca.html)

#### Metriken:

Die Metriken für alle Zufallswerte werden in [`outputs/kmeans_tfidf/kmeans_tfidf_all_runs.json`](outputs/kmeans_tfidf/kmeans_tfidf_all_runs.json) gespeichert. Die Details zum besten Lauf stehen zusätzlich in [`outputs/kmeans_tfidf/best_kmeans_tfidf_summary.json`](outputs/kmeans_tfidf/best_kmeans_tfidf_summary.json). Für den aktuellen besten Lauf ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.07025986164808273 | Cluster sind nur schwach getrennt |
| Davies–Bouldin Index | 1.911063822888633 | mittlere Überlappung zwischen den Clustern |
| Calinski–Harabasz Index | 2.129976708816034 | schwache Clusterstruktur |

#### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF-Raum; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung. Es wurde die Gruppierung des besten Seeds interpretiert.

| Cluster | Top-Wörter |
| --- | --- |
| 0 | perfusion, hsi, measurements, literature, article, clinical, studies, noninvasive, modality, performed |
| 1 | disease, technology, information, biological, data, tissue, high, disorders, resolution, spectral imaging |
| 2 | medical, hsi, medical applications, research, machine, learning, study, diagnosis, field, future |
| 3 | hsi, tissue, brain, different, capabilities, vision, systems, surgical, technologies, spatial |
| 4 | spectroscopy, light, use, color, techniques, based, modalities, compared, skin, surgery |
| 5 | learning, medical, images, algorithms, principles, systems, data, image, visualization, studies |
| 6 | systems, hsi, vivo, patients, studies, current, tissue, systematic review, emerging, guidance |
| 7 | patients, studies, detection, small, meta analysis, meta, lesions, improvement, results, literature |
| 8 | multispectral, lesions, skin, multispectral imaging, level, advances, tissue, technique, allows, summarize |
| 9 | cancer, hsi, accuracy, computer aided, aided, computer, detection, sensitivity, diagnostic, studies |

### Evaluation

Die aktuelle Konfiguration ist der Referenzstand des Experiments. Die Aktivierung der englischen Stopwords hat die Tokenqualität verbessert, weil sehr allgemeine Wörter weniger stark in die Darstellung eingehen. Dadurch werden die Cluster-Terme inhaltlich klarer lesbar.

---

## dbscan + tfidf

### Kurzüberblick

- **Kurzbeschreibung:** TF‑IDF-Feature-Extraktion gefolgt von DBSCAN-Clustering. Ziel ist, dichte, thematische Gruppen ohne feste Clusteranzahl zu identifizieren.

### Konfiguration

Die Experimentkonfiguration liegt in [configs/dbscan_tfidf.yaml](configs/dbscan_tfidf.yaml).

```yaml
experiment_name: dbscan_tfidf

input:
  documents_path: data/raw/data_db_raw.csv
  format: csv
  text_fields: [title, abstract]
  fuse_mode: join
  separator: ";"

dbscan:
  eps: 0.75
  min_samples: 3
  metric: cosine
  leaf_size: 30
  p: null
  n_jobs: null

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
  output_dir: outputs/dbscan_tfidf
  plot_name: dbscan_tfidf_pca.png
  summary_name: best_dbscan_tfidf_summary.json
  point_size: 42
  alpha: 0.85
  figsize_width: 10
  figsize_height: 7
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/tfidf.py`
3. Clustering mit `src/clustering/dbscan.py`
4. Evaluation mit `src/evaluation/basic_unsupervised.py`
5. Outputs: PNG-Plot und Summary-JSON im Unterordner unter `outputs/dbscan_tfidf/` speichern

### Ergebnisse

#### Plot:

![dbscan + tfidf PCA](outputs/dbscan_tfidf/dbscan_tfidf_pca.png)

Eine interaktive Version die im Browser geöffnet werden muss befinet sich hier: [outputs/dbscan_tfidf/dbscan_tfidf_pca.html](outputs/dbscan_tfidf/dbscan_tfidf_pca.html)

#### Metriken: Die in der JSON gespeicherten Kennzahlen direkt auswerten.

Die Metriken werden in `outputs/dbscan_tfidf/best_dbscan_tfidf_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | 0.026386065408587456 | Cluster kaum getrennt |
| Davies–Bouldin Index | 2.1547427614491355 | mittlere Überlappung zwischen den Clustern |
| Calinski–Harabasz Index | 1.1843962957162295 | schwache Clusterstruktur |

#### Cluster-Interpretation

Die folgende Tabelle zeigt die wichtigsten Terme je Cluster aus der aktuellen Interpretation. Die Wörter stammen aus dem nicht reduzierten TF‑IDF-Raum; die zugehörigen Gewichte stehen in der JSON-Zusammenfassung.

| Cluster | Top-Wörter |
| --- | --- |
| -1 | light, color, high, spectra, approach, resolution, skin, using, compared, challenges |
| 0 | medical, tissue, studies, technology, cancer, clinical, disease, multispectral, detection, systems |


### Evaluation
Die Kennzahlen zeigen, dass die gefundene Clusterstruktur sehr schwach ausgeprägt ist. Ebenfalls würde mit den zwei gefundenen Clustern kaum Information gewonnen. DBSCAN ist deutlich durch die geringen Anzahl an Datenpunkten und damit einer folglich geringen Dichte beeinträchtigt. Es sollte auf eine großere Datenbasis angewendet werden.
Ebenfalls kann eine Optimierung der Hyperparameter noch hinzugefügt werden.

---

## Template für weitere Experimente

## <experiment_name>

### Kurzüberblick

- **Kurzbeschreibung:** kurze, natürliche Beschreibung des Experiments und der Zielsetzung.

### Konfiguration

Die Experimentkonfiguration liegt in [configs/<experiment>.yaml]().

```yaml
experiment_name: <experiment_name>

file body
```

### Pipeline

1. Daten einlesen (`data/raw/`)
2. Feature-Extraktion mit `src/features/<feature_extractor>.py`
3. Clustering mit `src/clustering/<algorithm>.py`
4. Evaluation mit `src/evaluation/<evaluator>.py`
5. Outputs: Plot und Summary im Unterordner unter `outputs/` speichern

### Ergebnisse

#### Plot:

![<experiment>]()


#### Metriken: Die in der JSON gespeicherten Kennzahlen direkt auswerten.

Die Metriken werden in `outputs/<experiment>/<experiment>_summary.json` gespeichert. Für das aktuelle Experiment ergibt sich:

| Metrik | Wert | Einordnung |
| --- | ---: | --- |
| Silhouette Score | <value> | <kurze Bewertung> |
| Davies–Bouldin Index | <value> | <kurze Bewertung> |
| Calinski–Harabasz Index | <value> | <kurze Bewertung> |

#### Cluster-Interpretation

### Evaluation

Kurze Interpretation der Ergebnisse und mögliche nächste Schritte. Hier kann auch direkt auf die Summary-JSON verwiesen werden.
