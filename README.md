<div align="center">

**[🇫🇷 Français](#fr)** · **[🇬🇧 English](#en)**

</div>

---

<a id="fr"></a>

# Customer Personality Analysis

**Segmenter les clients d'une épicerie en groupes homogènes pour adapter chaque campagne marketing au lieu de parler à tout le monde de la même façon.**

## En bref

- **Problème** : une épicerie envoie les mêmes messages à tous ses clients, alors qu'ils n'ont ni les mêmes moyens ni les mêmes habitudes.
- **Données** : 2 240 fiches clients (revenu, âge, famille, dépenses par produit, canaux d'achat, réponses aux campagnes), 2 208 conservées après nettoyage.
- **Méthode** : regroupement automatique des clients qui se ressemblent (apprentissage non supervisé), en comparant 3 algorithmes et en codant nous-mêmes l'algorithme principal.
- **Résultat** : 4 profils clients clairs (premium, fidèles, familles à budget modéré, jeunes connectés), chacun avec une piste d'action marketing.

## Sommaire

[Contexte](#contexte-et-problématique) · [Données](#données) · [Méthode](#méthode) · [Résultats](#résultats) · [Limites](#limites-et-pistes-damélioration) · [Structure](#structure-du-repo) · [Installation](#installation-et-exécution) · [Équipe](#équipe) · [Stack](#stack-technique)

## Contexte et problématique

Une épicerie de quartier veut mieux connaître sa clientèle pour arrêter de communiquer de la même manière auprès de tout le monde. Il n'existe aucune étiquette « type de client » dans les données : il faut donc **découvrir** les groupes à partir des comportements, puis décrire chaque groupe de façon exploitable par une équipe marketing.

Le projet se déroule en deux temps :

1. **Iris (cas d'école)** : un dataset dont les 3 classes sont connues sert à valider notre propre implémentation de K-Means, comparée à celle de scikit-learn.
2. **Épicerie (cas réel)** : la même démarche est appliquée, sans vérité terrain, aux données clients.

## Données

| | Iris | Épicerie |
|---|---|---|
| Source | `sklearn.datasets.load_iris` | `data/raw/marketing_campaign.csv` (dataset public Kaggle [*Customer Personality Analysis*](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)) |
| Taille brute | 150 fleurs × 4 mesures | 2 240 clients × 29 colonnes |
| Après nettoyage | 149 (1 doublon retiré) | 2 208 clients |

**Principales variables de l'épicerie** : année de naissance, niveau d'études, statut marital, revenu, enfants et adolescents au foyer, date d'inscription, jours depuis le dernier achat, dépenses par catégorie (vin, fruits, viande, poisson, sucreries, or), nombre d'achats par canal (promotions, web, catalogue, magasin), visites web par mois, réponses aux 6 campagnes marketing (`AcceptedCmp1` à `AcceptedCmp5`, `Response`), réclamations.

## Méthode

**1. Nettoyage** (2 240 → 2 208 clients)
- 24 clients sans revenu supprimés ;
- 4 statuts maritaux aberrants supprimés (« Absurd », « YOLO ») ;
- 3 clients nés avant 1930 et 1 revenu supérieur à 200 000 retirés comme valeurs aberrantes.

**2. Exploration** : statistiques descriptives, distributions et matrice de corrélation. Celle-ci montre que les dépenses par produit et les achats par canal sont très corrélés entre eux.

**3. Feature engineering et préparation**
- variables créées : âge (calculé à la date d'inscription la plus récente du dataset), ancienneté client en jours, dépenses totales, nombre total d'achats, nombre total de campagnes acceptées, nombre d'enfants ;
- regroupement des modalités : études en 3 niveaux, statut marital en « en couple » / « seul » ;
- encodage one-hot puis standardisation (31 colonnes).

**4. Réduction de dimension**
- sélection de variables : les colonnes détaillées, redondantes avec les agrégats, sont retirées, il en reste **13** ;
- ACP à **3 composantes**, qui conservent **47,1 %** de la variance (26,0 % + 11,2 % + 9,9 %).

**5. Choix du nombre de groupes** : méthode du coude et score de silhouette pour k = 2 à 7.

**6. Modèles comparés** (même k, mêmes données)
- **K-Means** (scikit-learn) ;
- **Classification ascendante hiérarchique** (CAH, critère de Ward), avec dendrogramme ;
- **Mélange de gaussiennes** (GMM).

**Choix final : K-Means avec 4 groupes.** C'est l'algorithme qui obtient le meilleur score de silhouette des trois. Le score seul favorise k = 2 (0,371), mais 2 groupes ne suffisent pas à différencier des campagnes. **k = 4** est retenu parce qu'il donne des segments interprétables et de tailles équilibrées.

**Validation sur Iris** : notre K-Means codé à la main (`kmeans.py` : algorithme de Lloyd, initialisation k-means++, 10 essais conservant la meilleure inertie) obtient **exactement la même inertie que scikit-learn (138,77)** et isole parfaitement les 50 *setosa*.

La veille technique (algorithmes, choix de K, mesures de qualité) est détaillée dans [`docs/veille.md`](docs/veille.md).

## Résultats

**En clair** : les 2 208 clients se répartissent en 4 groupes de taille comparable (entre 499 et 657 clients). Chaque groupe a un niveau de revenu, une façon d'acheter et une réceptivité aux campagnes qui lui sont propres. L'épicerie peut donc adresser à chacun un message différent.

### Les 4 profils clients

| Profil | Clients | Ce qui le caractérise | Action marketing suggérée par l'analyse |
|---|---|---|---|
| **Premium** | 531 | Revenu le plus élevé, dépenses de loin les plus fortes, presque pas d'enfants, le groupe qui accepte le plus les campagnes | Cible prioritaire des offres premium |
| **Fidèles** | 521 | Revenu confortable, dépenses élevées, le plus grand nombre d'achats et la plus longue ancienneté | Segment stable et engagé, à fidéliser |
| **Familles à budget modéré** | 657 | Revenu moyen, le plus d'enfants au foyer, peu d'achats et des dépenses limitées | Offres familiales ou promotions |
| **Jeunes connectés** | 499 | Les plus jeunes et le revenu le plus faible, les dépenses les plus basses, mais le plus de visites sur le site web | Convertir progressivement vers plus de dépenses |

### Détail technique

**Comparaison des algorithmes** (k = 4, sur les 3 composantes de l'ACP)

| Algorithme | Silhouette | Tailles des groupes |
|---|---|---|
| **K-Means** (retenu) | **0,285** | 657 / 499 / 531 / 521 |
| GMM | 0,256 | 528 / 593 / 512 / 575 |
| CAH (Ward) | 0,226 | 801 / 478 / 459 / 470 |

**Choix de k (K-Means)**

| k | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| Inertie | 7 864,82 | 6 355,92 | 5 188,51 | 4 538,20 | 3 982,26 | 3 648,51 |
| Silhouette | 0,371 | 0,291 | 0,285 | 0,273 | 0,271 | 0,266 |

**Profil moyen par groupe (K-Means)**

| Profil | Revenu | Âge | Dépenses totales | Achats totaux | Enfants | Ancienneté (jours) | Campagnes acceptées | Visites web / mois |
|---|---|---|---|---|---|---|---|---|
| Premium | 77 800,6 | 45,3 | 1 336,7 | 20,4 | 0,2 | 300,4 | 1,0 | 2,4 |
| Fidèles | 59 072,0 | 49,3 | 894,9 | 22,1 | 1,1 | 509,3 | 0,5 | 6,0 |
| Familles à budget modéré | 41 883,3 | 47,8 | 158,5 | 9,7 | 1,4 | 243,2 | 0,1 | 5,8 |
| Jeunes connectés | 30 231,1 | 37,0 | 120,0 | 8,3 | 1,0 | 396,9 | 0,2 | 7,1 |

**Iris (validation)** : inertie de 138,77 pour notre implémentation comme pour scikit-learn. Silhouette de 0,581 à k = 2 et de 0,461 à k = 3. *Versicolor* et *virginica* se chevauchent partiellement.

## Limites et pistes d'amélioration

**Limites**
- **Séparation modeste** : une silhouette de 0,285 indique des groupes qui se chevauchent, ce qui est fréquent sur des données clients réelles (sur Iris, le score atteint 0,58).
- **Choix de k** : k = 4 est un choix d'interprétabilité métier, le score de silhouette seul désignant k = 2.
- **Information perdue** : l'ACP à 3 composantes ne conserve que 47,1 % de l'information.
- **Variables mélangées** : les variables catégorielles encodées sont mélangées aux variables numériques dans une même distance euclidienne.
- **Pas de test terrain** : les actions marketing sont des pistes issues du profil moyen de chaque groupe. Elles n'ont pas été testées sur le terrain.
- **Module non exécuté** : le module de métriques de Kylliann ([`travaux/kylliann/`](travaux/kylliann/)) n'a pas encore été exécuté sur les données de l'épicerie.

**Pistes**
- tester d'autres nombres de composantes, ou une analyse factorielle multiple (AFM) adaptée au mélange quantitatif / catégoriel ;
- essayer DBSCAN pour isoler les clients atypiques ;
- pondérer les variables selon leur importance métier (revenu, dépenses) ;
- compléter l'évaluation avec les indices de Davies-Bouldin, de Calinski-Harabasz et la statistique Gap, déjà codés dans `travaux/kylliann/kmeans.py`.

## Structure du repo

| Fichier / dossier | Rôle |
|---|---|
| [`notebook.ipynb`](notebook.ipynb) | Notebook principal : Iris (validation), puis épicerie (nettoyage, exploration, ACP, 3 algorithmes, profiling) |
| [`kmeans.py`](kmeans.py) | K-Means codé à la main (Lloyd, k-means++, `n_init`), utilisé par le notebook |
| [`data/raw/marketing_campaign.csv`](data/raw/marketing_campaign.csv) | Données brutes de l'épicerie (séparateur tabulation) |
| [`docs/veille.md`](docs/veille.md) | Veille technique : algorithmes, choix de K, mesures de qualité |
| [`docs/images/`](docs/images/) | Illustrations de la veille |
| [`travaux/kylliann/`](travaux/kylliann/) | Module de Kylliann : K-Means et métriques de choix de K en NumPy pur, avec son notebook de démonstration |
| `requirements.txt` | Dépendances (pip) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dépendances (uv), Python 3.10 |

## Installation et exécution

Avec **pip** (Python 3.10 ou plus) :

```bash
git clone https://github.com/axel-achart/customer-personality-analysis.git
cd customer-personality-analysis
python -m venv .venv
# Windows : .venv\Scripts\activate    |    macOS / Linux : source .venv/bin/activate
pip install -r requirements.txt

python kmeans.py                 # test rapide : notre K-Means vs scikit-learn sur Iris
jupyter notebook notebook.ipynb  # notebook principal
```

Avec **uv** :

```bash
uv sync
uv run python kmeans.py
uv run --with jupyter jupyter notebook notebook.ipynb
```

Le notebook lit `data/raw/marketing_campaign.csv` en chemin relatif : il faut le lancer depuis la racine du repo.

## Équipe

| Membre | GitHub | Contributions principales |
|---|---|---|
| Axel Achart | [axel-achart](https://github.com/axel-achart) | Notebook Iris et épicerie, K-Means codé à la main, veille (synthèse) |
| Hippolyte Geslain | [Hippolyte-Geslain](https://github.com/Hippolyte-Geslain) | Lancement du projet, veille, première version de la fonction K-Means, intégration sur `develop` |
| Kylliann Larcher | [kylliann-larcher](https://github.com/kylliann-larcher) | Veille (algorithmes, schémas, sources), module de métriques en NumPy pur (`travaux/kylliann/`) |

Projet réalisé à **LaPlateforme_** (Marseille), dans le cadre du **Bachelor IA / Data**, en juin et juillet 2026.

## Stack technique

Python 3.10 · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Jupyter · uv

<p align="right"><a href="#fr">↑ Haut de page</a> · <a href="#en">🇬🇧 English version</a></p>

---

<a id="en"></a>

# Customer Personality Analysis

**Segmenting a grocery store's customers into homogeneous groups so each marketing campaign can be tailored, instead of sending the same message to everyone.**

## At a glance

- **Problem**: a grocery store sends the same messages to all its customers, even though they have neither the same budget nor the same habits.
- **Data**: 2,240 customer records (income, age, household, spending per product, purchase channels, campaign responses), 2,208 kept after cleaning.
- **Approach**: automatically grouping similar customers (unsupervised learning), comparing 3 algorithms and coding the main one ourselves.
- **Outcome**: 4 clear customer profiles (premium, loyal, budget-conscious families, young digital-first), each with a marketing lead.

## Contents

[Context](#context-and-problem) · [Data](#data) · [Approach](#approach) · [Results](#results) · [Limitations](#limitations-and-next-steps) · [Structure](#repository-structure) · [Installation](#installation-and-usage) · [Team](#team) · [Tech stack](#tech-stack)

## Context and problem

A neighbourhood grocery store wants to know its customers better and stop addressing everyone the same way. The data contains no "customer type" label, so the groups have to be **discovered** from behaviour, then described in a way a marketing team can act on.

The project has two stages:

1. **Iris (warm-up)**: a dataset with 3 known classes, used to validate our own K-Means implementation against scikit-learn's.
2. **Grocery store (real case)**: the same approach applied to customer data, with no ground truth.

## Data

| | Iris | Grocery store |
|---|---|---|
| Source | `sklearn.datasets.load_iris` | `data/raw/marketing_campaign.csv` (public Kaggle dataset [*Customer Personality Analysis*](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)) |
| Raw size | 150 flowers × 4 measurements | 2,240 customers × 29 columns |
| After cleaning | 149 (1 duplicate removed) | 2,208 customers |

**Main grocery variables**: year of birth, education, marital status, income, children and teenagers at home, sign-up date, days since last purchase, spending per category (wine, fruit, meat, fish, sweets, gold), number of purchases per channel (deals, web, catalogue, store), monthly web visits, responses to the 6 marketing campaigns (`AcceptedCmp1` to `AcceptedCmp5`, `Response`), complaints.

## Approach

**1. Cleaning** (2,240 → 2,208 customers)
- 24 customers with no income removed;
- 4 invalid marital statuses removed ("Absurd", "YOLO");
- 3 customers born before 1930 and 1 income above 200,000 removed as outliers.

**2. Exploration**: descriptive statistics, distributions and a correlation matrix. It shows that spending per product and purchases per channel are strongly correlated with each other.

**3. Feature engineering and preparation**
- new variables: age (computed at the most recent sign-up date in the dataset), customer tenure in days, total spending, total purchases, total accepted campaigns, number of children;
- category grouping: education into 3 levels, marital status into "partner" / "single";
- one-hot encoding, then standardisation (31 columns).

**4. Dimensionality reduction**
- feature selection: detailed columns that duplicate the aggregates are dropped, leaving **13**;
- PCA with **3 components**, keeping **47.1%** of the variance (26.0% + 11.2% + 9.9%).

**5. Number of groups**: elbow method and silhouette score for k = 2 to 7.

**6. Models compared** (same k, same data)
- **K-Means** (scikit-learn);
- **Agglomerative hierarchical clustering** (Ward linkage), with a dendrogram;
- **Gaussian Mixture Model** (GMM).

**Final choice: K-Means with 4 groups.** It gets the best silhouette score of the three. The score alone favours k = 2 (0.371), but 2 groups are not enough to differentiate campaigns. **k = 4** was chosen because it gives interpretable, balanced segments.

**Validation on Iris**: our hand-coded K-Means (`kmeans.py`: Lloyd's algorithm, k-means++ initialisation, 10 runs keeping the best inertia) reaches **exactly the same inertia as scikit-learn (138.77)** and perfectly isolates the 50 *setosa*.

The technical research notes (algorithms, choosing K, quality metrics) are in [`docs/veille.md`](docs/veille.md) (in French).

## Results

**In plain terms**: the 2,208 customers split into 4 groups of similar size (between 499 and 657 customers). Each group has its own income level, buying pattern and response to campaigns, so the store can send each one a different message.

### The 4 customer profiles

| Profile | Customers | Key traits | Marketing action suggested by the analysis |
|---|---|---|---|
| **Premium** | 531 | Highest income, by far the highest spending, almost no children, most responsive to campaigns | Priority target for premium offers |
| **Loyal** | 521 | Comfortable income, high spending, most purchases and longest tenure | Stable, engaged segment to retain |
| **Budget-conscious families** | 657 | Average income, most children at home, few purchases and limited spending | Family offers or promotions |
| **Young digital-first** | 499 | Youngest and lowest income, lowest spending, but the most website visits | Gradually convert to higher spending |

### Technical details

**Algorithm comparison** (k = 4, on the 3 PCA components)

| Algorithm | Silhouette | Group sizes |
|---|---|---|
| **K-Means** (selected) | **0.285** | 657 / 499 / 531 / 521 |
| GMM | 0.256 | 528 / 593 / 512 / 575 |
| Agglomerative (Ward) | 0.226 | 801 / 478 / 459 / 470 |

**Choosing k (K-Means)**

| k | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| Inertia | 7,864.82 | 6,355.92 | 5,188.51 | 4,538.20 | 3,982.26 | 3,648.51 |
| Silhouette | 0.371 | 0.291 | 0.285 | 0.273 | 0.271 | 0.266 |

**Average profile per group (K-Means)**

| Profile | Income | Age | Total spending | Total purchases | Children | Tenure (days) | Accepted campaigns | Web visits / month |
|---|---|---|---|---|---|---|---|---|
| Premium | 77,800.6 | 45.3 | 1,336.7 | 20.4 | 0.2 | 300.4 | 1.0 | 2.4 |
| Loyal | 59,072.0 | 49.3 | 894.9 | 22.1 | 1.1 | 509.3 | 0.5 | 6.0 |
| Budget-conscious families | 41,883.3 | 47.8 | 158.5 | 9.7 | 1.4 | 243.2 | 0.1 | 5.8 |
| Young digital-first | 30,231.1 | 37.0 | 120.0 | 8.3 | 1.0 | 396.9 | 0.2 | 7.1 |

**Iris (validation)**: an inertia of 138.77 for both our implementation and scikit-learn. Silhouette of 0.581 at k = 2 and 0.461 at k = 3. *Versicolor* and *virginica* partly overlap.

## Limitations and next steps

**Limitations**
- **Modest separation**: a silhouette of 0.285 means the groups overlap, which is common on real customer data (Iris reaches 0.58).
- **Choice of k**: k = 4 is a business-interpretability choice; the silhouette score alone points to k = 2.
- **Lost information**: the 3-component PCA keeps only 47.1% of the information.
- **Mixed variables**: encoded categorical variables are mixed with numeric ones in a single Euclidean distance.
- **Not field-tested**: the marketing actions are leads derived from each group's average profile. They have not been tested in practice.
- **Module not yet run**: Kylliann's metrics module ([`travaux/kylliann/`](travaux/kylliann/)) has not yet been run on the grocery data.

**Next steps**
- try other numbers of components, or a Multiple Factor Analysis (MFA) suited to mixed numeric / categorical data;
- try DBSCAN to isolate atypical customers;
- weight variables by business importance (income, spending);
- extend the evaluation with the Davies-Bouldin and Calinski-Harabasz indices and the Gap statistic, already coded in `travaux/kylliann/kmeans.py`.

## Repository structure

| File / folder | Role |
|---|---|
| [`notebook.ipynb`](notebook.ipynb) | Main notebook: Iris (validation), then grocery store (cleaning, exploration, PCA, 3 algorithms, profiling) |
| [`kmeans.py`](kmeans.py) | Hand-coded K-Means (Lloyd, k-means++, `n_init`), used by the notebook |
| [`data/raw/marketing_campaign.csv`](data/raw/marketing_campaign.csv) | Raw grocery data (tab-separated) |
| [`docs/veille.md`](docs/veille.md) | Technical research notes (French): algorithms, choosing K, quality metrics |
| [`docs/images/`](docs/images/) | Illustrations for the research notes |
| [`travaux/kylliann/`](travaux/kylliann/) | Kylliann's module: pure-NumPy K-Means and K-selection metrics, with its demo notebook |
| `requirements.txt` | Dependencies (pip) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies (uv), Python 3.10 |

## Installation and usage

With **pip** (Python 3.10 or later):

```bash
git clone https://github.com/axel-achart/customer-personality-analysis.git
cd customer-personality-analysis
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

python kmeans.py                 # quick check: our K-Means vs scikit-learn on Iris
jupyter notebook notebook.ipynb  # main notebook
```

With **uv**:

```bash
uv sync
uv run python kmeans.py
uv run --with jupyter jupyter notebook notebook.ipynb
```

The notebook reads `data/raw/marketing_campaign.csv` with a relative path, so run it from the repository root.

## Team

| Member | GitHub | Main contributions |
|---|---|---|
| Axel Achart | [axel-achart](https://github.com/axel-achart) | Iris and grocery notebook, hand-coded K-Means, research notes (summary) |
| Hippolyte Geslain | [Hippolyte-Geslain](https://github.com/Hippolyte-Geslain) | Project kick-off, research notes, first K-Means function, integration on `develop` |
| Kylliann Larcher | [kylliann-larcher](https://github.com/kylliann-larcher) | Research notes (algorithms, diagrams, sources), pure-NumPy metrics module (`travaux/kylliann/`) |

School project at **LaPlateforme_** (Marseille), **Bachelor in AI / Data**, June–July 2026.

## Tech stack

Python 3.10 · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Jupyter · uv

<p align="right"><a href="#en">↑ Back to top</a> · <a href="#fr">🇫🇷 Version française</a></p>
