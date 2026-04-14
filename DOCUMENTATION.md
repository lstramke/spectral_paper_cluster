# Dokumentation

Es folgt eine Erläuterung der verwendeteten Metriken, CLuster Algorithmen und Feature Extraktoren, die in den Experimenten verwendet werden. Die genauen Experiment-Konfigurationen und Ergebnisse sind in den jeweiligen Experimentordnern zu finden, diese werden teilweise verlinkt sein. Ansonsten siehe [README](README.md).

## Metriken

Zur Bewertung der Clusterqualität werden drei etablierte Kennzahlen verwendet. Sie liefern Hinweise darauf, ob die resultierenden Gruppen intern kompakt, untereinander getrennt und damit statistisch wie inhaltlich plausibel sind.
Die Kennzahlen werden im Experiment aus den Clusterlabels und Feature-Vektoren mit Funktionen aus `sklearn.metrics` berechnet.

## **Silhouette Score**

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

Dabei ist $a(i)$ der durchschnittliche Abstand eines Punkts zu den anderen Punkten im eigenen Cluster und $b(i)$ der kleinste durchschnittliche Abstand zu einem anderen Cluster. Der Wert liegt zwischen $-1$ und $1$. In der Praxis gelten Werte über etwa $0.5$ oft als brauchbar bis gut, Werte nahe $0$ als Hinweis auf überlappende Cluster und negative Werte als deutliches Warnsignal für Fehlzuordnungen. Werte nahe $1$ sprechen für eine sehr saubere Trennung.

## **Davies–Bouldin Index**

$$
DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \ne i} \frac{S_i + S_j}{M_{ij}}
$$

Hier beschreibt $S_i$ die Streuung innerhalb von Cluster $i$ und $M_{ij}$ den Abstand zwischen den Zentren der Cluster $i$ und $j$. Der Index ist nach unten besser: kleine Werte bedeuten kompakte Cluster mit gutem Abstand zueinander. In vielen Anwendungen sind Werte unter etwa $1$ bereits recht ordentlich, während Werte deutlich über $2$ oder $3$ häufig auf überlappende oder instabile Cluster hindeuten.

## **Calinski–Harabasz Index**

$$
CH = \frac{\mathrm{Tr}(B_k)/(k-1)}{\mathrm{Tr}(W_k)/(n-k)}
$$

Dabei steht $B_k$ für die Streuung zwischen den Clustern und $W_k$ für die Streuung innerhalb der Cluster. Höhere Werte sind besser, weil sie bedeuten, dass die Cluster intern kompakt und untereinander gut getrennt sind. Absolutwerte sind allerdings stark von Datenmenge, Dimensionalität und Clusteranzahl abhängig; wichtig ist daher vor allem der relative Vergleich zwischen verschiedenen Einstellungen. Größere Werte sind in der Regel besser, sehr kleine Werte deuten auf eine schwache Clusterstruktur hin.

Zusammenfassend gilt: Für eine gute Clusterstruktur sollten Silhouette und Calinski–Harabasz eher hoch sein, während der Davies–Bouldin Index eher niedrig sein sollte. Praktisch spricht das für kompakte, gut getrennte Cluster; niedrige oder unklare Werte deuten dagegen auf überlappende oder schwach ausgeprägte Gruppen hin. Gemeinsam liefern die Kennzahlen so eine verständliche Einschätzung der Clusterqualität.

## Cluster‑Algorithmen

Die folgenden Abschnitte beschreiben die in diesem Repository genutzten Clustering‑Algorithmen. Die Implementierungen stammen jeweils aus `sklearn` und wurden über Adapter in die Pipeline integriert. Die Algorithmusauswahl erfolgte in dieser Phase bewusst explorativ: Statt vorab einen vermeintlich "besten" Standardalgorithmus festzulegen, wurden mehrere Verfahren systematisch unter vergleichbaren Bedingungen getestet. Auf Basis dieser Ergebnisse soll die Auswahl für zukünftige Feature‑Extraktoren gezielt auf die jeweils geeignetsten Verfahren eingegrenzt werden.

### K‑Means

K‑Means ist ein centroidenbasiertes Verfahren. Es ordnet jeden Punkt dem nächstgelegenen Zentrum zu und aktualisiert danach die Zentren iterativ. Das zugehörige Optimierungsproblem lautet:
$$
\min_{c,\mu} J(c,\mu) = \sum_{i=1}^{n} \left\| x_i - \mu_{c_i} \right\|^2
$$

**Ablauf (Pseudocode):**
```text
Algorithm: K-Means Clustering
Input: data matrix X, cluster count k, max iterations T_max
Output: assignments c, centers μ

1:  μ <- InitializeCenters(X, k)
2:  for t <- 1 to T_max do
3:      for each point x_i do
4:          cᵢ <- argminⱼ ‖xᵢ − μⱼ‖²
5:      end for
6:      for each cluster j in {1..k} do
7:          Cⱼ <- {xᵢ ∣ cᵢ = j}
8:          μⱼ⁽ⁿᵉʷ⁾ <- (1 / |Cⱼ|) · ∑ₓᵢ∈Cⱼ xᵢ
9:      end for
10:    Jₜ <- ∑ᵢ ‖xᵢ − μ_{cᵢ}‖²
11:    if t > 1 and |Jₜ − Jₜ₋₁| < ε then
12:        break
13:    end if
14:    μ <- μ⁽ⁿᵉʷ⁾
15: end for
16: return (c, μ)
```

- **Stärken:** Einfach, schnell, gut skalierbar; funktioniert gut bei kompakten, kugelförmigen Clustern.
- **Schwächen:** Benötigt festes `k`; sensitiv gegenüber Initialisierung, Ausreißern und nicht-konvexen Strukturen.

- Implementierung: [src/clustering/kmeans.py](src/clustering/kmeans.py)
- Verwendet in: [experiments/kmeans_tfidf/kmeans_tfidf.md](experiments/kmeans_tfidf/kmeans_tfidf.md)

### DBSCAN

DBSCAN ist dichtebasiert und definiert Cluster über Nachbarschaften mit Radius $\varepsilon$ und Mindestanzahl `minPts`.

$$
N_{\varepsilon}(x) = \{y \mid d(x,y) \le \varepsilon\}
$$

Ein Punkt ist Kernpunkt, wenn $|N_{\varepsilon}(x)| \geq \text{minPts}$. Cluster entstehen über dichte‑erreichbare Punkte, übrige Punkte werden als Rauschen markiert.

**Ablauf (Pseudocode):**
```text
Algorithm: DBSCAN
Input: points X, radius ε, minimum neighbors minPts
Output: labels y

1:  y <- InitializeLabels(|X|, UNASSIGNED)
2:  currentClusterId <- 0
3:  for each point p in X do
4:      if Visited(p) then
5:          continue
6:      end if
7:      MarkVisited(p)
8:      N <- {q ∈ X ∣ d(p,q) ≤ ε}
9:      if |N| < minPts then
10:         y[p] <- NOISE
11:     else
12:         currentClusterId <- currentClusterId + 1
13:         y[p] <- currentClusterId
14:         seedSet <- N
15:         while seedSet not empty do
16:             q <- pop(seedSet)
17:             if not Visited(q) then
18:                 MarkVisited(q)
19:                 N_q <- {u ∈ X ∣ d(q,u) ≤ ε}
20:                 if |N_q| ≥ minPts then
21:                     seedSet <- seedSet union N_q
22:                 end if
23:             end if
24:             if y[q] in {UNASSIGNED, NOISE} then
25:                 y[q] <- currentClusterId
26:             end if
27:         end while
28:     end if
29: end for
30: return y
```

- **Stärken:** Findet Cluster beliebiger Form; erkennt Rauschen explizit; kein fixes `k` erforderlich.
- **Schwächen:** Parameterwahl ($\varepsilon$, `minPts`) ist sensitiv; Probleme bei stark variierenden Dichten.

- Implementierung: [src/clustering/dbscan.py](src/clustering/dbscan.py)
- Verwendet in: [experiments/dbscan_tfidf/dbscan_tfidf.md](experiments/dbscan_tfidf/dbscan_tfidf.md)

### OPTICS

OPTICS arbeitet ebenfalls dichtebasiert, erzeugt aber eine geordnete Reachability‑Struktur statt nur eines festen Schnitts.

$$
\begin{aligned}
\mathrm{core\_dist}_{\varepsilon,\mathrm{MinPts}}(p) &= \text{Distanz zum MinPts-ten Nachbarn von } p \\
\mathrm{reachability\_dist}(p \rightarrow o) &= \max\left(\mathrm{core\_dist}(p), d(p,o)\right)
\end{aligned}
$$

Die beiden Größen steuern die Reihenfolge, in der Punkte als "gut erreichbar" betrachtet werden: Die Core-Distanz beschreibt die lokale Dichte um einen Punkt (klein = dichtes Umfeld), die Reachability-Distanz kombiniert diese Dichteschwelle mit dem direkten Abstand zwischen zwei Punkten. Aus dem Reachability‑Plot lassen sich danach Cluster unterschiedlicher Dichte extrahieren.

**Ablauf (Pseudocode):**
```text
Algorithm: OPTICS (Reachability Ordering)
Input: points X, radius ε, minimum neighbors minPts
Output: ordered points O, reachability distances, derived clusters

1:  O <- empty list
2:  Q <- priority queue ordered by reachability distance
3:  for each point p in X do
4:      if Visited(p) then
5:          continue
6:      end if
7:      MarkVisited(p)
8:      Nₚ <- {q ∈ X ∣ d(p,q) ≤ ε}
9:      coreDist(p) <- distance to minPts-th nearest neighbor in N_p
10:    append p to O
11:    if coreDist(p) is defined then
12:        for each o in Nₚ do
13:            newReach <- max(coreDist(p), d(p,o))
14:            if reachability(o) undefined or newReach < reachability(o) then
15:                reachability(o) <- newReach
16:                push/update o in Q by reachability(o)
17:            end if
18:        end for
19:    end if
20:    while Q not empty do
21:        q <- extract-min(Q)
22:        if Visited(q) then continue end if
23:        MarkVisited(q)
24:        N_q <- {u ∈ X ∣ d(q,u) ≤ ε}
25:        coreDist(q) <- distance to minPts-th nearest neighbor in N_q
26:        append q to O
27:        if coreDist(q) is defined then
28:            update neighbors of q with new reachability values
29:        end if
30:    end while
10: end for
11: return ExtractClustersFromReachability(O, xi)
```

- **Stärken:** Robuster als DBSCAN bei unterschiedlichen Dichten; liefert reichhaltige Reachability-Struktur.
- **Schwächen:** Komplexer zu interpretieren; Clusterextraktion (z. B. `xi`) benötigt zusätzliche Entscheidungen.

- Implementierung: [src/clustering/optics.py](src/clustering/optics.py)
- Verwendet in: [experiments/optics_tfidf/optics_tfidf.md](experiments/optics_tfidf/optics_tfidf.md)

### HDBSCAN

HDBSCAN erweitert DBSCAN hierarchisch über Dichteschwellen und extrahiert stabile Cluster.

Statt ein einziges globales $\varepsilon$ zu verwenden, betrachtet HDBSCAN eine Hierarchie über Dichtelevel (typisch über $\lambda = 1/d$) und wählt daraus stabile Cluster.
Intuitiv gilt: `coreDist` beschreibt die lokale Dichteschwelle, `mrd` (mutual reachability distance) macht paarweise Abstände unter dieser Schwelle vergleichbar, und `stability` misst, wie lange ein Cluster über Dichtelevel hinweg bestehen bleibt.

**Ablauf (Pseudocode):**
```text
Algorithm: HDBSCAN (simplified)
Input: points X, minimum cluster size minClusterSize
Output: labels y

1:  for each point x do
2:      coreDistₖ(x) <- d(x, kNNₖ(x))   # k = minSamples
3:  end for
4:  for each edge (a,b) do
5:      mrd(a,b) <- max(coreDistₖ(a), coreDistₖ(b), d(a,b))
6:  end for
7:  M <- minimum spanning tree with edge weights mrd
8:  T <- hierarchy induced by removing MST edges from high to low density
9:  T_condensed <- condense T by minClusterSize
10: for each cluster C in T_condensed do
11:     stability(C) <- ∑ₓ∈C (λ_death(x) − λ_birth(C))
12: end for
13: C_selected <- select non-overlapping clusters with highest stability
14: y <- assign points to C_selected; remaining points -> NOISE
15: return y
```

- **Stärken:** Kein globales $\varepsilon$; gute Behandlung variierender Dichten; stabile Cluster durch Persistenzkriterium.
- **Schwächen:** Höhere algorithmische Komplexität; bei kleinen Datensätzen oft viele Punkte als Rauschen.

- Implementierung: [src/clustering/hdbscan.py](src/clustering/hdbscan.py)
- Verwendet in: [experiments/hdbscan_tfidf/hdbscan_tfidf.md](experiments/hdbscan_tfidf/hdbscan_tfidf.md)

### Agglomerative Clustering

Agglomerative Clusterung startet mit Einzelpunkten und verschmilzt iterativ Cluster nach Linkage‑Kriterium.

Für Average‑Linkage gilt beispielsweise:

$$
d(A,B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a,b)
$$

Das Dendrogramm wird anschließend über `distance_threshold` oder `n_clusters` geschnitten.
Intuitiv steuert das Linkage-Kriterium, welche Cluster als Nächstes zusammengeführt werden: `single` bevorzugt lokale Verbindungen, `complete` eher kompakte Cluster, `average` liegt dazwischen.

**Ablauf (Pseudocode):**
```text
Algorithm: Agglomerative Clustering
Input: points X, linkageType, targetClusters k or distanceThreshold τ
Output: cluster set C

1:  C <- InitializeSingletonClusters(X)
2:  while not StopCriterion(C, k, tau) do
3:      for each cluster pair (A, B) in C do
4:          if linkageType = average then
5:              d(A,B) <- (1 / (|A||B|)) · ∑ₐ∈A ∑ᵦ∈B d(a,b)
6:          else if linkageType = complete then
7:              d(A,B) <- maxₐ∈A,ᵦ∈B d(a,b)
8:          else if linkageType = single then
9:              d(A,B) <- minₐ∈A,ᵦ∈B d(a,b)
10:        end if
11:     end for
12:     (A, B) <- argmin_{cluster pairs} d(A,B)
13:     M <- A union B
14:     C <- (C \ {A, B}) union {M}
15: end while
16: return C
```

- **Stärken:** Kein Stochasticity-Thema durch Initialisierung; Dendrogramm erlaubt Analyse auf mehreren Granularitäten.
- **Schwächen:** Rechenintensiver bei vielen Punkten; frühe Fehl-Merges sind irreversibel.

- Implementierung: [src/clustering/agglomerativeClustering.py](src/clustering/agglomerativeClustering.py)
- Verwendet in: [experiments/agglomerative_tfidf/agglomerative_tfidf.md](experiments/agglomerative_tfidf/agglomerative_tfidf.md)

### Affinity Propagation

Affinity Propagation sucht Exemplare über Nachrichtenaustausch zwischen Punkten (Responsibility und Availability).

$$
\begin{aligned}
r(i,k) &= s(i,k) - \max_{k' \neq k} \{a(i,k') + s(i,k')\} \\
a(i,k) &= \min\left(0, r(k,k) + \sum_{i' \notin \{i,k\}} \max(0, r(i',k))\right)
\end{aligned}
$$

Dabei bezeichnet $i$ den aktuell betrachteten Datenpunkt und $k$ einen potenziellen Exemplar-Kandidaten. $k'$ steht für alternative Kandidaten im Maximalvergleich und $i'$ für andere Datenpunkte in der Summation.

Der Parameter `preference` beeinflusst direkt die Anzahl gefundener Exemplare/Cluster.
Intuitiv gilt: `r(i,k)` (responsibility) bewertet, wie gut Punkt `k` als Exemplar für `i` geeignet ist, während `a(i,k)` (availability) beschreibt, wie stark `k` als Exemplar für `i` unterstützt wird.

**Ablauf (Pseudocode):**
```text
Algorithm: Affinity Propagation
Input: similarity matrix S, damping α, max iterations T_max
Output: exemplar indices E

1:  R <- InitializeMessageMatrix(|S|, |S|)   # responsibilities
2:  A <- InitializeMessageMatrix(|S|, |S|)   # availabilities
3:  for t <- 1 to T_max do
4:      for all i,k do
5:          rₙₑw(i,k) <- s(i,k) − max_{k' ≠ k}(a(i,k') + s(i,k'))
6:      end for
7:      for all i != k do
8:          aₙₑw(i,k) <- min(0, r(k,k) + ∑_{i' not in {i,k}} max(0, r(i',k)))
9:      end for
10:    for all k do
11:        aₙₑw(k,k) <- ∑_{i' ≠ k} max(0, r(i',k))
12:    end for
13:    R <- α · R + (1 - α) · rₙₑw
14:    A <- α · A + (1 - α) · aₙₑw
15:    if MessagesConverged(R, A) then
16:        break
17:    end if
18: end for
19: E <- {k ∣ a(k,k) + r(k,k) > 0}
20: for each point i do
21:     label(i) <- argmaxₖ (a(i,k) + r(i,k))
22: end for
23: return E, label
```

- **Stärken:** Keine feste Clusteranzahl nötig; exemplar-basierte Interpretation oft intuitiv.
- **Schwächen:** Sensitiv gegenüber `preference` und Dämpfung; quadratischer Speicher-/Zeitbedarf in der Punktanzahl.

- Implementierung: [src/clustering/affinityPropagation.py](src/clustering/affinityPropagation.py)
- Verwendet in: [experiments/affinityPropagation_tfidf/affinityPropagation_tfidf.md](experiments/affinityPropagation_tfidf/affinityPropagation_tfidf.md)

### Spectral Clustering

Spectral Clustering bildet einen Ähnlichkeitsgraphen und nutzt dessen Laplace‑Spektrum zur Einbettung vor dem finalen Clustering.

Dabei ist $W$ die Affinitätsmatrix und $D$ die Gradmatrix; der (unnormierte) Graph-Laplacian ist $L = D - W$. Die kleinsten Eigenvektoren von $L$ liefern eine Repräsentation, in der anschließend typischerweise k‑means angewandt wird.
Dabei bezeichnen $i$ und $j$ zwei Datenpunkte im Graphen; $\gamma$ steuert bei RBF die Nachbarschaftsschärfe (größerer Wert betont lokale Ähnlichkeiten stärker), während bei kNN die Nachbarschaft über die nächsten Nachbarn diskret festgelegt wird.
Intuitiv entsteht so eine Einbettung, in der Punkte aus derselben Struktur näher zusammenliegen, bevor die finale Clusterzuordnung erfolgt.

**Ablauf (Pseudocode):**
```text
Algorithm: Spectral Clustering
Input: points X, cluster count k, affinityType
Output: labels y

1:  for each pair (i,j) do
2:      if affinityType = rbf then
3:          W_ij <- exp(-γ * ||x_i - x_j||^2)
4:      else if affinityType = knn then
5:          W_ij <- 1 if x_j ∈ kNN(x_i), else 0
6:      end if
7:  end for
8:  Dᵢᵢ <- ∑ⱼ Wᵢⱼ
9:  L <- D - W
10: U <- eigenvectors of L for k smallest eigenvalues
11: normalize each row of U (optional but common)
12: (y, _) <- KMeans(U, k, 300)
13: return y
```

- **Stärken:** Sehr gut für nicht-konvexe Clusterformen; Graphsicht auf lokale Nachbarschaften.
- **Schwächen:** Eigenzerlegung kann teuer sein; empfindlich gegenüber Wahl von Affinität, $\gamma$, `n_neighbors`.

- Implementierung: [src/clustering/spectralClustering.py](src/clustering/spectralClustering.py)
- Verwendet in: [experiments/spectral_tfidf/spectral_tfidf.md](experiments/spectral_tfidf/spectral_tfidf.md)

## Feature‑Extractor

Die folgenden Abschnitte beschreiben die in diesem Repository eingesetzten Feature‑Extraktoren. Feature‑Extraktion ist dabei der zentrale Schritt, um unstrukturierte Texte in numerische Repräsentationen zu überführen, auf denen Clustering-Verfahren überhaupt sinnvoll arbeiten können. Aktuell ist TF‑IDF  mit LSA als robuste und gut interpretierbare Basis implementiert.

Perspektivisch ist geplant, weitere Repräsentationen zu ergänzen und die Auswahl der Feature‑Extraktoren auf Basis der bisherigen Experimente sowie der Datencharakteristik gezielt zu erweitern.

### TF‑IDF (mit optionaler LSA)

TF‑IDF gewichtet Tokens nach Häufigkeit im Dokument und Seltenheit im Korpus.

$$
\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d) \cdot \log\left(\frac{N}{\mathrm{df}(t)}\right)
$$

Optional wird per LSA (Truncated SVD) reduziert:

$$
X \approx U_k \Sigma_k V_k^T
$$

Dadurch werden semantische Hauptachsen extrahiert und die Dimension reduziert.

**Ablauf (Pseudocode):**
```text
Algorithm: TF-IDF with optional LSA reduction
Input: documents D, boolean useLsa, component count k_lsa
Output: feature matrix X

1:  T <- TokenizeDocuments(D)
2:  V <- BuildVocabulary(T)
3:  for each document d and term t do
4:      tf(t,d) <- count(t,d) / ∑_{t'} count(t',d)
5:  end for
6:  for each term t do
7:      df(t) <- number of documents containing t
8:      idf(t) <- log((N + 1) / (df(t) + 1)) + 1
9:  end for
10: for each document d and term t do
11:     X[d,t] <- tf(t,d) * idf(t)
12: end for
13: if useLsa then
14:     X <- TruncatedSVD(X, k_lsa)   # X ≈ U_k Σ_k V_k^T
15: end if
16: return X
```

- **Stärken:** Sehr interpretierbare Features; robust als Baseline; mit LSA bessere Kompaktheit und weniger Rauschen.
- **Schwächen:** Bag-of-Words ignoriert Wortreihenfolge/Kontext; großes Vokabular kann hochdimensional und spärlich sein.

- Implementierung: [src/features/tfidf.py](src/features/tfidf.py)
- Verwendet in: 
	- [experiments/kmeans_tfidf/kmeans_tfidf.md](experiments/kmeans_tfidf/kmeans_tfidf.md)
	- [experiments/dbscan_tfidf/dbscan_tfidf.md](experiments/dbscan_tfidf/dbscan_tfidf.md)
	- [experiments/optics_tfidf/optics_tfidf.md](experiments/optics_tfidf/optics_tfidf.md)
	- [experiments/hdbscan_tfidf/hdbscan_tfidf.md](experiments/hdbscan_tfidf/hdbscan_tfidf.md)
	- [experiments/agglomerative_tfidf/agglomerative_tfidf.md](experiments/agglomerative_tfidf/agglomerative_tfidf.md)
	- [experiments/affinityPropagation_tfidf/affinityPropagation_tfidf.md](experiments/affinityPropagation_tfidf/affinityPropagation_tfidf.md)
	- [experiments/spectral_tfidf/spectral_tfidf.md](experiments/spectral_tfidf/spectral_tfidf.md)
