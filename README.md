# customer-personality-analysis
"Statistics show that statistics cannot be trusted."

---

## Veille technique

### Regroupement non supervisé

Le regroupement non supervisé, aussi appelé classification non supervisée, consiste à rassembler des données non étiquetées en groupes homogènes à partir de leurs similitudes, sans qu'un modèle ait été préalablement entraîné sur des exemples connus.

Source : [Google Cloud](https://cloud.google.com/discover/supervised-vs-unsupervised-learning?hl=fr)

---

### Clustering

Le clustering est une technique de machine learning non supervisée conçue pour regrouper des exemples non étiquetés en fonction de leur similarité. Il permet d'identifier des tendances et des structures cachées dans les données.

Source : [Google ML Glossary](https://developers.google.com/machine-learning/clustering/overview?hl=fr)

---

### Algorithmes de classification non supervisée

La classification non supervisée regroupe les données sans étiquettes connues à l'avance. Les algorithmes se répartissent en plusieurs familles :

**- Par partitionnement (centroïdes)**

**K-Means** : regroupe les points en K groupes selon leur proximité au centroïde de chaque cluster. C'est la méthode de partitionnement la plus répandue en machine learning.

**- Par hiérarchie (dendrogramme CAH)**

**Clustering hiérarchique ascendant (agglomératif)** : chaque exemple démarre dans son propre cluster, puis les clusters les plus proches sont fusionnés itérativement pour former un arbre hiérarchique.

**Clustering hiérarchique descendant (divisif)** : tous les exemples sont regroupés en un seul cluster, puis divisés itérativement.

**- Par densité**

**DBSCAN** : identifie des clusters en se basant sur la densité des points, ce qui permet de détecter des formes arbitraires et de gérer efficacement les points de bruit.

**OPTICS** : extension de DBSCAN qui gère mieux les clusters de densités variables.

---

### Fonctionnement de 3 algorithmes

**- K-Means**

L'algorithme initialise K centroïdes (aléatoirement ou issus du dataset), puis itère jusqu'à convergence :

1. Chaque point est assigné au centroïde le plus proche.
2. Les centroïdes sont recalculés comme la moyenne des points du cluster.
3. On répète jusqu'à stabilisation.

Avantage : simple, rapide et scalable. Limite : K doit être fixé à l'avance ; sensible aux outliers et aux formes non convexes.

**- Clustering hiérarchique ascendant**

Chaque point démarre dans son propre cluster. On calcule la matrice de ressemblance entre tous les couples, puis on fusionne itérativement les deux clusters les plus proches. Le résultat est un dendrogramme que l'on coupe à la hauteur voulue pour obtenir le nombre de clusters souhaité.

Avantage : pas besoin de fixer K a priori ; visualisation intuitive. Limite : coûteux en mémoire et en calcul sur de grands datasets.

**- DBSCAN**

Deux paramètres gouvernent l'algorithme : ε (distance maximale entre deux points voisins) et MinPts (nombre minimum de points pour former un cluster dense).

1. Un point avec au moins MinPts voisins dans un rayon ε est un point central (core point).
2. L'algorithme explore les voisins et ajoute les points éligibles au cluster.
3. Les points n'appartenant à aucun cluster sont marqués comme bruit (noise).

Avantage : détecte des formes quelconques, isole les outliers naturellement, ne nécessite pas de fixer K. Limite : peu efficace quand les clusters ont des densités variables ; difficile à paramétrer en haute dimension.

---

### Méthodes de sélection du nombre optimal de clusters

**Méthode du coude (Elbow Method)** : méthode graphique qui trace l'inertie en fonction du nombre de clusters K. On recherche le point d'inflexion (le "coude") où l'ajout d'un cluster supplémentaire n'apporte plus de gain significatif.
Source : [IBM](https://www.ibm.com/fr-fr/think/topics/k-means-clustering)

**Score de silhouette** : mesure la similarité d'un point avec son propre cluster par rapport aux autres clusters. Plus le score est élevé, meilleur est le clustering.
Source : [Medium – therised](https://therised.medium.com/determining-the-number-of-clusters-a-comprehensive-guide-1a2441c5a526)

**Statistiques d'écart (Gap Statistics)** : compare les performances du clustering sur les données réelles à celles obtenues sur des données aléatoires, afin de déterminer si la structure identifiée est réelle.
Source : [Medium – therised](https://therised.medium.com/determining-the-number-of-clusters-a-comprehensive-guide-1a2441c5a526)

---

### Mesures de qualité d'un cluster

Les métriques internes évaluent la qualité du clustering sans étiquettes de référence. Ce sont les seules disponibles dans un contexte vraiment non supervisé.

**Silhouette Score**

Pour chaque point i, on calcule a(i) (distance moyenne au sein de son cluster) et b(i) (distance moyenne au cluster le plus proche). Le score est : `s(i) = (b(i) - a(i)) / max(a(i), b(i))`, et varie de −1 à +1.

| Score | Interprétation |
|---|---|
| Proche de +1 | Bien clusterisé |
| Autour de 0 | Proche d'une frontière |
| Proche de −1 | Probablement mal classé |

Source : Peter Rousseeuw, 1987 (Wikipedia / arXiv)




---
**Indice Davies-Bouldin (DBI)**

Mesure la similarité moyenne entre chaque cluster et son plus proche voisin, en comparant le rapport dispersion interne / distance entre centroïdes. Une valeur plus basse indique une meilleure qualité de clustering.
Source : scikit-learn docs / GeeksforGeeks / arXiv

**Inertie (Within-Cluster Sum of Squares)**

Somme des distances au carré entre chaque point et le centroïde de son cluster. Une forte inertie inter-classes traduit une bonne séparation des clusters, tandis qu'une faible inertie intra-classe traduit leur compacité. Utilisée notamment pour la méthode du coude.
Source : HAL

**Calinski-Harabasz Index**

Mesure le rapport entre la variance inter-clusters et la variance intra-cluster. Un score plus élevé signifie des clusters compacts et bien séparés. Cet indice aide à déterminer le nombre idéal de clusters.
Source : GeeksforGeeks

---