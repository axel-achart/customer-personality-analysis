# customer-personality-analysis
*“Statistics show that statistics cannot be trusted.”*

Le projet va suivre les objectifs pédagogiques de la certification simpleton et ceux de la plateforme.
Le projet sera analysé et entrepris par les différents biais professionnels de chacun, ce projet va montrer notre personnalité professionnelle et accompagner la montée en compétences transversale scolaire et professionnelle.

Uv utilisé pour initialiser le projet

### Objectifs pédagogiques :

**BC01 - Données**
Automatiser l'extraction de données -> récupération
du dataset de l'épicerie
Développer des requêtes SQL -> exploration et
interrogation des données
Développer des règles d'agrégation -> nettoyage,
suppression des entrées corrompues,
homogénéisation des formats (étape b et c du projet)

**BC02- Modeles & Services IA**
Organiser et réaliser une veille technique -> veille
explicitement demandee sur les algorithmes de clustering et
les methodes de selection du nombre optimal de clusters
Monitorer un modèle d'IA -> évaluation de la qualité des
clusters, comparaison des resultats entre plusieurs valeurs
de k et plusieurs algorithmes
Programmer les tests automatisés d'un modèle d'IA ->tests
du modele K-means pour plusieurs valeurs de k, evaluation
des métriques

**BC03- Application IA**
Analyser le besoin -> comprehension de la problematique
nétier (segmentation client)
Concevoir le cadre technique -> choix des algorithmes,
rchitecture du notebook, sélection de features et réduction de
dimension (MFA)
Coordonner la realisation technique -> gestion de projet via
Trello, organisation agile
Developper les composants techniques -> implementation de la
lasse K-means custom en Python (kmeans.py), notebooks
upyter
Automatiser les phases de tests -> integration continue du code
ersionne sur GitHub
Surveiller une application d'IA -> profiling des groupes,
nterprétation des clusters, feedback loop
Resoudre les incidents techniques - debogage, documentation
des solutions dans le README

**Première tournée de décisions**

La veille sera structurée en 3 composants : 
- Titre de la notion
- Sources de la veille
- Description longue de la notion

La première réunion portera sur la mise en commun du contexte métier et technique préliminaire.

Veille :

Classification automatique non supervisée
Algorithmes :

- K-means
On intervient et on choisit n points pour n groupes obtenus à la fin de la classification
pro : rapide
con : si mauvais nombre alors 

- CaH, classification ascendante hiérarchique
pro : explorer visuellement sans utiliser "k"

- dbscan
si zone assez dense, création de clusters 
pro : redéfinit outliers comme bruit
con : ne marche pas pour un groupe hétérogène

- algorithme apriori pour aller plus loin 

Lors de l'analyse des persona, il faut bien chercher à trouver les critères classiques :  [chercher les critères utilisés dans les persona de vente d'articles]


mesures du nombre de clusters : 

méthode du coude 
méthode gap statistic

mesure de la qualité de cluster : 

méthode silouhette 