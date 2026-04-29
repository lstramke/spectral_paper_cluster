# Experimentzusammenfassung

Tabelle zum Vergleich der Experimente nach Cluster-Anzahl und Metriken. Jede Zeile verlinkt auf die jeweilige Summary-Datei im `experiments/`-Ordner.

## Dataset: results_41

### Extractor: fasttext

| Experiment | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown |
| --- | ---: | ---: | ---: | ---: | --- |
| dbscan_fasttext | 1 |  |  |  | [summary](experiments/dbscan_fasttext/results_41/dbscan_fasttext_41.md) |


### Extractor: tfidf

| Experiment | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown |
| --- | ---: | ---: | ---: | ---: | --- |
| affinityPropagation_tfidf | 9 | 0.0911928117275238 | 2.0933003434378477 | 2.054876870442089 | [summary](experiments/affinityPropagation_tfidf/results_41/affinityPropagation_tfidf_41.md) |
| agglomerative_tfidf | 10 | 0.14200261235237122 | 1.8365325829568917 | 2.148901278585822 | [summary](experiments/agglomerative_tfidf/results_41/agglomerative_tfidf_41.md) |
| dbscan_tfidf | 1 | 0.04912794381380081 | 2.152774204441197 | 1.1866975846121282 | [summary](experiments/dbscan_tfidf/results_41/dbscan_tfidf_41.md) |
| gaussianMixture_tfidf | 10 | 0.08469785004854202 | 2.0553832063634028 | 2.0330562688400016 | [summary](experiments/gaussianMixture_tfidf/results_41/gaussianMixture_tfidf_41.md) |
| hdbscan_tfidf | 4 | 0.01987781934440136 | 2.8439736011699557 | 1.839823045321821 | [summary](experiments/hdbscan_tfidf/results_41/hdbscan_tfidf_41.md) |
| kmeans_tfidf | 10 | 0.1251147836446762 | 1.9727481809192724 | 2.064641326951606 | [summary](experiments/kmeans_tfidf/results_41/kmeans_tfidf_41.md) |
| optics_tfidf | 2 | 0.07049638032913208 | 2.9323449516790725 | 2.521684262709035 | [summary](experiments/optics_tfidf/results_41/optics_tfidf_41.md) |
| spectral_tfidf | 10 | 0.11248621344566345 | 1.9227023029672683 | 2.0304430584237467 | [summary](experiments/spectral_tfidf/results_41/spectral_tfidf_41.md) |


## Dataset: results_2086

### Extractor: bert

| Experiment | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown |
| --- | ---: | ---: | ---: | ---: | --- |
| affinityPropagation_bert | 51 | 0.5981753468513489 | 0.8424029267128099 | 1105.4158109244113 | [summary](experiments/affinityPropagation_bert/results_2086/affinityPropagation_bert_2086.md) |
| agglomerative_bert | 38 | 0.569751501083374 | 0.7862829404617879 | 988.8531553139046 | [summary](experiments/agglomerative_bert/results_2086/agglomerative_bert_2086.md) |
| hdbscan_bert | 48 | 0.37247234582901 | 1.1351390551385243 | 244.56703266872208 | [summary](experiments/hdbscan_bert/results_2086/hdbscan_bert_2086.md) |
| kmeans_bert | 38 | 0.6127510666847229 | 0.8124082779279529 | 1089.5667690752662 | [summary](experiments/kmeans_bert/results_2086/kmeans_bert_2086.md) |
| spectral_bert | 40 | 0.5634604096412659 | 0.7942174123046718 | 1017.0879020479808 | [summary](experiments/spectral_bert/results_2086/spectral_bert_2086.md) |


### Extractor: fasttext

| Experiment | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown |
| --- | ---: | ---: | ---: | ---: | --- |
| affinityPropagation_fasttext | 123 | 0.040174368768930435 | 2.437045763918542 | 15.264215122324721 | [summary](experiments/affinityPropagation_fasttext/results_2086/affinityPropagation_fasttext_2086.md) |
| agglomerative_fasttext | 2 | 0.69720059633255 | 0.9863880926760953 | 13.250217858074196 | [summary](experiments/agglomerative_fasttext/results_2086/agglomerative_fasttext_2086.md) |
| dbscan_fasttext | 1 | 0.892043948173523 | 0.3388814357842144 | 16.66710516056775 | [summary](experiments/dbscan_fasttext/results_2086/dbscan_fasttext_2086.md) |
| gaussianMixture_fasttext | 5 | 0.1393960565328598 | 2.9197658266087227 | 151.957795087624 | [summary](experiments/gaussianMixture_fasttext/results_2086/gaussianMixture_fasttext_2086.md) |
| hdbscan_fasttext | 2 | 0.30456462502479553 | 3.467138268349481 | 27.697547769836014 | [summary](experiments/hdbscan_fasttext/results_2086/hdbscan_fasttext_2086.md) |
| kmeans_fasttext | 5 | 0.16747154295444489 | 2.0845118570592236 | 143.1266675616661 | [summary](experiments/kmeans_fasttext/results_2086/kmeans_fasttext_2086.md) |
| spectral_fasttext | 5 | 0.15155474841594696 | 2.4350888802150257 | 137.48326678570024 | [summary](experiments/spectral_fasttext/results_2086/spectral_fasttext_2086.md) |


### Extractor: tfidf

| Experiment | n_clusters | silhouette | davies_bouldin | calinski_harabasz | summary markdown |
| --- | ---: | ---: | ---: | ---: | --- |
| affinityPropagation_tfidf | 176 | 0.26424601674079895 | 1.7716755566247075 | 16.440219700130623 | [summary](experiments/affinityPropagation_tfidf/results_2086/affinityPropagation_tfidf_2086.md) |
| agglomerative_tfidf | 132 | 0.2717430889606476 | 1.6730642750088283 | 18.07832378325674 | [summary](experiments/agglomerative_tfidf/results_2086/agglomerative_tfidf_2086.md) |
| dbscan_tfidf | 19 | 0.06773672997951508 | 2.1881803095409063 | 21.951147621359112 | [summary](experiments/dbscan_tfidf/results_2086/dbscan_tfidf_2086.md) |
| gaussianMixture_tfidf | 20 | 0.13955402374267578 | 3.0673331208113805 | 33.24359931241326 | [summary](experiments/gaussianMixture_tfidf/results_2086/gaussianMixture_tfidf_2086.md) |
| hdbscan_tfidf | 73 | 0.08024097979068756 | 2.1040121387844115 | 14.443256265588882 | [summary](experiments/hdbscan_tfidf/results_2086/hdbscan_tfidf_2086.md) |
| kmeans_tfidf | 40 | 0.20143121480941772 | 2.620221613949932 | 25.640215288652055 | [summary](experiments/kmeans_tfidf/results_2086/kmeans_tfidf_2086.md) |
| optics_tfidf | 1 | 0.05994307994842529 | 1.2141481885763068 | 15.62638434689137 | [summary](experiments/optics_tfidf/results_2086/optics_tfidf_2086.md) |
| spectral_tfidf | 40 | 0.1702127307653427 | 2.521106186145267 | 23.460122799765603 | [summary](experiments/spectral_tfidf/results_2086/spectral_tfidf_2086.md) |

