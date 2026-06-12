### Un algorithme de classification non supervisée aussi appelé apprentissage non supervié consiste à laisser un algorithme trouver tout seul des groupes dans les données sans étiquettes connues à l'avance.

### Processus:

```mermaid
flowchart LR
	A[1. Donnees brutes] --> B[2. L'algorithme repere des similarites]
	B --> C[3. Formation de segments ou clusters]
```
### Algorithme de classification non supervisée

#### K-Means

Le K-Means est un algorithme de clustering non supervisé qui regroupe automatiquement les données en K clusters selon leur similarité en minimisant la distance entre chaque point et le centre de son cluster.

#### Schéma de fonctionnement (étape par étape)

```mermaid
flowchart TD
	A[1. Choisir K clusters] --> B[2. Initialiser K centroïdes]
	B --> C[3. Assigner chaque point au centroïde le plus proche]
	C --> D[4. Recalculer les centroïdes]
	D --> E{5. Convergence atteinte ?}
	E -- Non --> C
	E -- Oui --> F[Clusters finaux]
```

#### Source:
1. Scikit-learn, Unsupervised learning (vue d’ensemble)  
https://scikit-learn.org/stable/unsupervised_learning.html

2. Scikit-learn, Clustering (inclut K-Means et son principe itératif)  
https://scikit-learn.org/stable/modules/clustering.html

3. Scikit-learn, KMeans API (paramètres, convergence, implémentation)  
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

4. MacQueen, J. (1967), Some Methods for Classification and Analysis of Multivariate Observations (papier fondateur de K-Means)  
https://projecteuclid.org/euclid.bsmsp/1200512992

5. Lloyd, S. (1982), Least Squares Quantization in PCM (algorithme de Lloyd souvent associé à K-Means)  
https://ieeexplore.ieee.org/document/1056489

6. Arthur, D. and Vassilvitskii, S. (2007), k-means++: The Advantages of Careful Seeding  
https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

#### La CAH (Classification Ascendante Hiérarchique)

Une méthode de clustering non supervisé qui regroupe progressivement les points les plus proches pour construire un dendrogramme, puis le coupe à un niveau choisi pour obtenir les clusters.

#### Schéma de fonctionnement (étape par étape)

```mermaid
flowchart TD
	A[1. Chaque point est un cluster initial] --> B[2. Calculer les distances entre clusters]
	B --> C[3. Fusionner les deux clusters les plus proches]
	C --> D[4. Mettre a jour les distances]
	D --> E{5. Nombre de clusters cible atteint ?}
	E -- Non --> C
	E -- Oui --> F[6. Couper le dendrogramme et obtenir les clusters finaux]
```

![alt text](image.png)

#### Sources CAH:
1. Scikit-learn, Clustering hiérarchique
https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

2. Scikit-learn, AgglomerativeClustering (API)
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

3. SciPy, cluster.hierarchy (linkage, dendrogram)
https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

4. Ward, J. H. (1963), Hierarchical Grouping to Optimize an Objective Function
https://doi.org/10.1080/01621459.1963.10500845

#### DBSCAN

Le DBSCAN est un algorithme de clustering non supervisé basé sur la densité, qui regroupe les points proches en clusters et identifie les points isolés comme du bruit.

#### Paramètres clés

1. eps (epsilon): rayon de voisinage autour d'un point.
2. min_samples: nombre minimum de points dans ce rayon pour qu'un point soit considéré comme un point coeur.

#### Schéma de fonctionnement (étape par étape)

```mermaid
flowchart TD
	A[1. Choisir eps et min_samples] --> B[2. Initialiser tous les points comme non visites]
	B --> C[3. Prendre un point non visite]
	C --> D{4. Voisins dans eps >= min_samples ?}
	D -- Oui --> E[5. Creer ou etendre un cluster]
	E --> F[6. Ajouter les points densite-connectes]
	F --> G{7. Reste-t-il des points non visites ?}
	G -- Oui --> C
	D -- Non --> H[5. Marquer comme bruit provisoire]
	H --> G
	G -- Non --> I[8. Clusters finaux + points bruit]
```

#### Sources DBSCAN:
1. Scikit-learn, DBSCAN (section Clustering)
https://scikit-learn.org/stable/modules/clustering.html#dbscan

2. Scikit-learn, DBSCAN API
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

3. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996), A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
https://cdn.aaai.org/KDD/1996/KDD96-037.pdf

4. Schubert, E. et al. (2017), DBSCAN Revisited, Revisited
https://doi.org/10.1145/3068335


### Quelles sont les méthodes de sélection du nombre optimal de clusters et la mesure de qualité d’un cluster.


L'idée centrale est simple : tu as des données, tu veux les regrouper automatiquement, mais tu ne sais pas en combien de groupes. Le clustering c'est ça — et les méthodes dont on parle servent à répondre à cette question.

On commence par le problème du choix de K :

**Méthode du coude** : on essaie K = 1, 2, 3, 4… et on mesure à chaque fois l'inertie (la somme des distances de chaque point à son centroïde). Naturellement l'inertie baisse quand K augmente — mais à partir d'un certain K, le gain devient minuscule. Ce "coude" indique le bon K.
![alt text](image-1.png)

**la méthode du score de silhouette**: c'est la plus fiable. Pour chaque point, on regarde deux choses : est-il proche des autres points de son groupe ? Et est-il loin du groupe voisin ? Si oui, son score est proche de +1. Si non (il est mal classé), son score est proche de -1.


![alt text](image-2.png)

Maintenant qu'on a des clusters, comment savoir s'ils sont bons ?

Il faut verifier 2 choses :
1. **Compacite** : les points d'un meme cluster sont proches entre eux.
2. **Separation** : les clusters sont bien eloignes les uns des autres.

### Comment fonctionnent les 3 methodes de mesure de qualite

1. **Silhouette**

Pour chaque point, on compare :
- sa distance moyenne a son propre cluster
- sa distance moyenne au cluster voisin le plus proche

Si le point est bien place, il est proche de son cluster et loin des autres.

Interpretation :
- proche de **1** : tres bon
- proche de **0** : clusters qui se chevauchent
- negatif : point souvent mal classe

Regle simple : **plus grand = mieux**.

2. **Davies-Bouldin**

On regarde si les clusters sont :
- compacts a l'interieur
- bien separes entre eux

Le score compare la dispersion interne des clusters a la distance entre leurs centres.

Regle simple : **plus petit = mieux**.

3. **Calinski-Harabasz**

Cette methode compare :
- la separation entre clusters
- la dispersion a l'interieur des clusters

Si les clusters sont loin les uns des autres et serres a l'interieur, le score augmente.

Regle simple : **plus grand = mieux**.

### Comment choisir K en pratique

1. Tester plusieurs valeurs de K (ex: 2 a 10).
2. Regarder la **methode du coude** pour trouver une zone plausible.
3. Verifier avec la **silhouette** (prendre un K avec un score eleve).
4. Garder un K interpretable metier (segments utiles pour l'analyse).

En resume :
- **Coude** sert surtout a choisir le nombre de clusters.
- **Silhouette / Davies-Bouldin / Calinski-Harabasz** servent a juger la qualite du clustering.

### Sources (selection de K et mesure de qualite)

1. Scikit-learn, KMeans (inertie / choix de K):
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

2. Scikit-learn, silhouette_score:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

3. Scikit-learn, davies_bouldin_score:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html

4. Scikit-learn, calinski_harabasz_score:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html

5. Rousseeuw, P. J. (1987), Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis:
https://www.sciencedirect.com/science/article/pii/0377042787901257

6. Davies, D. L. and Bouldin, D. W. (1979), A Cluster Separation Measure:
https://ieeexplore.ieee.org/document/4766909

7. Calinski, T. and Harabasz, J. (1974), A Dendrite Method for Cluster Analysis:
https://www.tandfonline.com/doi/abs/10.1080/03610927408827101

