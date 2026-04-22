# Experimentzusammenfassung

Tabelle zum Vergleich der Experimente nach Cluster-Anzahl und Metriken. Jede Zeile verlinkt auf die jeweilige Summary-Datei im `experiments/`-Ordner.

## Dataset: results_41

| Experiment                | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown                                                                            |
| ------------------------- | ---------: | ---------: | -------------: | ----------------: | ------------------------------------------------------------------------------------------- |
| spectral_tfidf            |         10 |  0.1124862 |      1.9227023 |         2.0304431 | [summary](experiments/spectral_tfidf/results_41/spectral_tfidf_41.md)                       |
| gaussianMixture_tfidf     |         10 | 0.08469785 |     2.05538321 |        2.03305627 | [summary](experiments/gaussianMixture_tfidf/results_41/gaussianMixture_tfidf_41.md)         |
| kmeans_tfidf              |         10 | 0.12511478 |     1.97274818 |        2.06464133 | [summary](experiments/kmeans_tfidf/results_41/kmeans_tfidf_41.md)                           |
| optics_tfidf              |          2 | 0.07049638 |     2.93234495 |        2.52168426 | [summary](experiments/optics_tfidf/results_41/optics_tfidf_41.md)                           |
| dbscan_tfidf              |          1 | 0.04912794 |     2.15277420 |        1.18669758 | [summary](experiments/dbscan_tfidf/results_41/dbscan_tfidf_41.md)                           |
| hdbscan_tfidf             |          4 | 0.01987782 |     2.84397360 |        1.83982305 | [summary](experiments/hdbscan_tfidf/results_41/hdbscan_tfidf_41.md)                         |
| affinityPropagation_tfidf |          9 | 0.09119281 |     2.09330034 |        2.05487687 | [summary](experiments/affinityPropagation_tfidf/results_41/affinityPropagation_tfidf_41.md) |
| agglomerative_tfidf       |         10 | 0.14200261 |     1.83653258 |        2.14890128 | [summary](experiments/agglomerative_tfidf/results_41/agglomerative_tfidf_41.md)             |

## Dataset: results_2086

| Experiment                | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown                                                                                |
| ------------------------- | ---------: | ---------: | -------------: | ----------------: | ----------------------------------------------------------------------------------------------- |
| spectral_tfidf            |         20 | 0.11032392 |     2.60374660 |       28.54055405 | [summary](experiments/spectral_tfidf/results_2086/spectral_tfidf_2086.md)                       |
| gaussianMixture_tfidf     |         20 | 0.13955402 |     3.06733312 |       33.24359931 | [summary](experiments/gaussianMixture_tfidf/results_2086/gaussianMixture_tfidf_2086.md)         |
| kmeans_tfidf              |         20 | 0.13751632 |     3.16378324 |       32.60649543 | [summary](experiments/kmeans_tfidf/results_2086/kmeans_tfidf_2086.md)                           |
| optics_tfidf              |          1 | 0.08690538 |     1.27885743 |       14.89671685 | [summary](experiments/optics_tfidf/results_2086/optics_tfidf_2086.md)                           |
| dbscan_tfidf              |          1 | 0.03307177 |     3.36243802 |        1.84101406 | [summary](experiments/dbscan_tfidf/results_2086/dbscan_tfidf_2086.md)                           |
| hdbscan_tfidf             |          9 | 0.01944141 |     3.47609566 |       22.91511778 | [summary](experiments/hdbscan_tfidf/results_2086/hdbscan_tfidf_2086.md)                         |
| affinityPropagation_tfidf |        169 | 0.16610596 |     2.16598802 |       12.37905130 | [summary](experiments/affinityPropagation_tfidf/results_2086/affinityPropagation_tfidf_2086.md) |
| agglomerative_tfidf       |        544 | 0.18110454 |     1.26399856 |        6.36906898 | [summary](experiments/agglomerative_tfidf/results_2086/agglomerative_tfidf_2086.md)             |
