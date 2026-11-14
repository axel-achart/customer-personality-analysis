# Module de Kylliann : métriques de choix de K (NumPy pur)

[← Retour au README](../../README.md)

> Travail de Kylliann Larcher, récupéré tel quel depuis la branche `feat/kylliann` (fichiers `kmeans.py` et `notebook.ipynb`, non modifiés).
> Ce notebook n'a **pas encore été exécuté** : il ne contient pas de sorties. Pour le lancer, se placer dans ce dossier (`cd travaux/kylliann`) afin que `from kmeans import ...` importe bien le module de ce dossier.


Toutes les métriques de sélection du K optimal et de mesure de qualité sont implémentées dans [`kmeans.py`](kmeans.py), en pur NumPy, sans dépendance à scikit-learn.

## Fonctions disponibles

| Fonction | Rôle | Règle d'interprétation |
|---|---|---|
| `find_optimal_k(x, max_k=10, ...)` | Teste K = 2 → max_k et retourne toutes les métriques + le K optimal selon chaque critère | — |
| `silhouette_score(x, labels)` | Score de silhouette moyen (entre -1 et 1) | **+ proche de 1 = meilleur** |
| `davies_bouldin_score(x, labels)` | Indice de Davies-Bouldin | **+ petit = meilleur** |
| `calinski_harabasz_score(x, labels)` | Indice de Calinski-Harabasz (Variance Ratio Criterion) | **+ grand = meilleur** |
| `gap_statistic(x, k, n_refs=10, ...)` | Statistique d'écart (Gap Statistic) | **+ grand = meilleur** |

## Utilisation rapide

```python
from kmeans import KMeans, find_optimal_k, silhouette_score, gap_statistic

# 1. Recherche automatique du K optimal
results = find_optimal_k(data, max_k=10, random_state=7)

print(f"K optimal (coude)            : {results['best_k_elbow']}")
print(f"K optimal (silhouette)       : {results['best_k_silhouette']}")
print(f"K optimal (Davies-Bouldin)   : {results['best_k_davies_bouldin']}")
print(f"K optimal (Calinski-Harabasz): {results['best_k_calinski_harabasz']}")
print(f"K optimal (Gap Statistic)    : {results['best_k_gap']}")

# 2. Entraînement avec le K retenu
model = KMeans(n_clusters=results["best_k_silhouette"], random_state=7)
model.fit(data)

# 3. Évaluation de la qualité du clustering obtenu
sil = silhouette_score(data, model.labels_)
gap = gap_statistic(data, k=results["best_k_silhouette"], random_state=7)
print(f"Silhouette finale : {sil:.4f}")
print(f"Gap Statistic     : {gap:.4f}")
```

## Structure de `find_optimal_k()`

```python
results = {
    "ks":                        [2, 3, 4, ...],   # K testés
    "inertias":                  [...],            # inertie pour chaque K
    "silhouette_scores":         [...],            # silhouette pour chaque K
    "davies_bouldin_scores":     [...],            # Davies-Bouldin pour chaque K
    "calinski_harabasz_scores":  [...],            # Calinski-Harabasz pour chaque K
    "gap_scores":                [...],            # Gap Statistic pour chaque K
    "best_k_elbow":              3,                # K optimal selon la méthode du coude
    "best_k_silhouette":         3,                # K optimal selon la silhouette
    "best_k_davies_bouldin":     3,                # K optimal selon Davies-Bouldin
    "best_k_calinski_harabasz":  3,                # K optimal selon Calinski-Harabasz
    "best_k_gap":                3,                # K optimal selon la Gap Statistic
}
```

## Détail de la méthode du coude

La fonction `_find_elbow_point()` utilise la **méthode du triangle** : pour chaque point de la courbe d'inertie, on calcule sa distance à la ligne reliant le premier point (K=1) au dernier point (K=max). Le K qui maximise cette distance est le coude.

Cette méthode est plus robuste qu'une simple inspection visuelle car elle fournit une détection **automatique et reproductible**.

## Détail de la Gap Statistic

La fonction `gap_statistic()` implémente la méthode de Tibshirani, Walther & Hastie (2001) :

1. Calcule l'inertie W_k sur les vraies données
2. Génère `n_refs` jeux de données de référence uniformément distribués dans le bounding-box des données (min/max par feature)
3. Applique K-means sur chaque jeu de référence pour obtenir W*_k
4. Gap(K) = moyenne(log(W*_k)) − log(W_k)

Un Gap élevé indique que les données réelles ont une structure de clusters bien plus marquée qu'une distribution aléatoire uniforme — donc que K est pertinent.

## Notebook de démonstration

Le notebook [`notebook.ipynb`](notebook.ipynb) contient une démonstration complète avec visualisations :
- Courbe du coude avec K=1 inclus
- Score de silhouette en fonction de K
- Indice de Davies-Bouldin en fonction de K
- Indice de Calinski-Harabasz en fonction de K
- Gap Statistic en fonction de K
- Grille de synthèse 3×2
- Clustering final avec les clusters colorés et les centroïdes
