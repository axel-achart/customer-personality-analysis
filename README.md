# customer-personality-analysis
"Statistics show that statistics cannot be trusted."

---

## Contexte du projet

Ce projet a pour but de segmenter la clientèle d'une épicerie de quartier par apprentissage non supervisé, afin de mieux comprendre qui sont ses clients et d'adapter ses actions marketing à chaque groupe plutôt que de communiquer de la même façon auprès de tout le monde.

Avant de travailler sur les données réelles de l'épicerie, le dataset **Iris** est utilisé comme cas d'école : ses 3 classes sont connues à l'avance, ce qui permet de valider notre propre implémentation de K-Means (`kmeans.py`) en la comparant à un clustering dont on connaît la "bonne réponse".

---

## Données

**Iris** (`sklearn.datasets.load_iris`) : 150 fleurs, 4 mesures (longueur/largeur des sépales et pétales) réparties en 3 espèces. Dataset propre, sans valeur manquante, utilisé uniquement pour valider les algorithmes.

**Épicerie** (`data/raw/marketing_campaign.csv`) : 2240 clients, 29 colonnes brutes (année de naissance, revenu, statut marital, nombre d'enfants, dépenses par catégorie de produit, nombre d'achats par canal, réponses aux campagnes marketing, etc.). Après nettoyage (valeurs manquantes sur le revenu, catégories aberrantes, quelques outliers d'âge/revenu), il reste 2208 clients. Plusieurs variables ont ensuite été recalculées par feature engineering (âge, ancienneté, dépenses totales, nombre d'achats total, nombre d'enfants, campagnes acceptées) pour remplacer des colonnes détaillées très corrélées entre elles. Le détail de l'exploration, du nettoyage et du feature engineering se trouve dans `notebook.ipynb`.

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

## Algorithmes utilisés

- **K-Means** : implémenté à la main dans `kmeans.py` (algorithme de Lloyd, initialisation k-means++, plusieurs essais gardés selon l'inertie), puis comparé à `sklearn.cluster.KMeans` sur Iris pour vérifier que les deux tombent sur le même résultat. Utilisé ensuite comme algorithme principal sur les données de l'épicerie (via scikit-learn).
- **Clustering hiérarchique ascendant (Agglomerative Clustering, linkage="ward")** : appliqué sur l'épicerie, avec un dendrogramme pour visualiser les regroupements successifs.
- **Gaussian Mixture Model (GMM)** : appliqué sur l'épicerie, pour comparer une approche probabiliste (mélange de gaussiennes) aux deux méthodes précédentes basées sur la distance.

Le nombre de clusters (k=3 pour Iris, k=4 pour l'épicerie) a été choisi via la méthode du coude et le score de silhouette. Les 3 algorithmes ont ensuite été comparés entre eux sur ce même k avec le score de silhouette, avant de profiler les clusters obtenus.

Sur l'épicerie, une réduction de dimension a été faite en amont : sélection de features pour retirer les variables détaillées redondantes (remplacées par les agrégats du feature engineering), puis PCA à 3 composantes (~47% de variance conservée) pour travailler sur un espace plus compact et plus facile à visualiser.

---

## Conclusion

Sur Iris, notre implémentation de K-Means donne exactement les mêmes résultats que celle de scikit-learn, ce qui valide l'algorithme. Le clustering retrouve bien la structure des 3 espèces, à l'exception de *versicolor* et *virginica* qui se chevauchent légèrement dans l'espace des variables.

Sur l'épicerie, la tâche est plus difficile : il n'y a pas de vraie étiquette à retrouver, et les scores de silhouette obtenus (~0.2-0.3) sont nettement plus modestes que sur Iris. K-Means est l'algorithme qui a le mieux performé parmi les 3 testés, devant le clustering hiérarchique et le GMM. Le profiling des 4 clusters fait ressortir des segments clients assez naturels : des clients premium à forte valeur, des clients fidèles à fréquence d'achat élevée, des familles à budget modéré, et des jeunes clients digital-first à faible pouvoir d'achat mais fort potentiel.

Le détail complet (visualisations, chiffres, interprétations) est dans `notebook.ipynb`. Pour la suite, plusieurs pistes restent à creuser : tester une MFA plutôt qu'une PCA classique pour mieux gérer le mélange de variables quantitatives et catégorielles, essayer DBSCAN pour isoler les clients atypiques, ou pondérer certaines variables (revenu, dépenses) selon leur importance métier avant de relancer le clustering.

---