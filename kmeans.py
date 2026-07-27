"""Implementation from scratch de K-means avec NumPy uniquement.

Objectif pédagogique : chaque étape de l'algorithme est explicite.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class KMeans:
	"""Classe K-means simple et bien commentée.

	Paramètres
	----------
	n_clusters : int, default=3
		Nombre de clusters (K) à construire.
	max_iter : int, default=300
		Nombre maximal d'itérations d'optimisation.
	tol : float, default=1e-4
		Seuil de convergence sur le déplacement des centroïdes.
	random_state : int | None, default=None
		Graine aléatoire pour la reproductibilité.
	init : str, default="k-means++"
		Stratégie d'initialisation : "random" ou "k-means++".
	"""

	def __init__(
		self,
		n_clusters: int = 3,
		max_iter: int = 300,
		tol: float = 1e-4,
		random_state: int | None = None,
		init: str = "k-means++",
	) -> None:
		if n_clusters <= 0:
			raise ValueError("n_clusters doit être strictement positif.")
		if max_iter <= 0:
			raise ValueError("max_iter doit être strictement positif.")
		if tol < 0:
			raise ValueError("tol doit être >= 0.")
		if init not in {"random", "k-means++"}:
			raise ValueError("init doit être 'random' ou 'k-means++'.")

		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self.init = init

		# Attributs renseignés après fit.
		self.cluster_centers_: np.ndarray | None = None
		self.labels_: np.ndarray | None = None
		self.inertia_: float | None = None
		self.n_iter_: int = 0

	def fit(self, x: np.ndarray) -> "KMeans":
		"""Entraîne K-means sur x.

		x a la forme (n_samples, n_features).
		"""
		x = self._validate_input(x)
		rng = np.random.default_rng(self.random_state)

		# Étape 1 : initialiser K centroïdes.
		centroids = self._initialize_centroids(x, rng)

		for iteration in range(1, self.max_iter + 1):
			# Étape 2 : assigner chaque point au centroïde le plus proche.
			labels = self._assign_labels(x, centroids)

			# Étape 3 : recalculer les centroïdes à partir des affectations.
			new_centroids = self._compute_centroids(x, labels, centroids, rng)

			# Test de convergence : si les centroïdes bougent peu, on arrête.
			max_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
			centroids = new_centroids

			if max_shift <= self.tol:
				self.n_iter_ = iteration
				break
		else:
			# Si la boucle n'a pas break, on a atteint max_iter.
			self.n_iter_ = self.max_iter

		# Affectation finale avec les centroïdes convergés.
		labels = self._assign_labels(x, centroids)

		self.cluster_centers_ = centroids
		self.labels_ = labels
		self.inertia_ = self._compute_inertia(x, labels, centroids)
		return self

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Assigne un cluster à de nouveaux points."""
		self._check_is_fitted()
		x = self._validate_input(x)
		return self._assign_labels(x, self.cluster_centers_)

	def fit_predict(self, x: np.ndarray) -> np.ndarray:
		"""Raccourci : fit puis retour des labels."""
		self.fit(x)
		return self.labels_

	def _initialize_centroids(
		self,
		x: np.ndarray,
		rng: np.random.Generator,
	) -> np.ndarray:
		if self.init == "random":
			return self._init_random(x, rng)
		return self._init_kmeans_plus_plus(x, rng)

	def _init_random(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
		"""Choisit K points uniques aléatoires comme centroïdes initiaux."""
		indices = rng.choice(x.shape[0], size=self.n_clusters, replace=False)
		return x[indices].copy()

	def _init_kmeans_plus_plus(
		self,
		x: np.ndarray,
		rng: np.random.Generator,
	) -> np.ndarray:
		"""Initialise les centroïdes avec k-means++ pour plus de stabilité."""
		n_samples = x.shape[0]
		centroids = np.empty((self.n_clusters, x.shape[1]), dtype=float)

		# Le premier centroïde est choisi aléatoirement.
		first_idx = rng.integers(0, n_samples)
		centroids[0] = x[first_idx]

		# Les centroïdes suivants sont choisis avec une probabilité
		# proportionnelle à la distance au carré au centroïde le plus proche.
		for i in range(1, self.n_clusters):
			distances_sq = np.min(
				np.sum((x[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
				axis=1,
			)

			total = distances_sq.sum()
			if total == 0:
				next_idx = rng.integers(0, n_samples)
			else:
				probabilities = distances_sq / total
				next_idx = rng.choice(n_samples, p=probabilities)

			centroids[i] = x[next_idx]

		return centroids

	@staticmethod
	def _assign_labels(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
		"""Retourne l'indice du centroïde le plus proche pour chaque point."""
		distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
		return np.argmin(distances, axis=1)

	def _compute_centroids(
		self,
		x: np.ndarray,
		labels: np.ndarray,
		previous_centroids: np.ndarray,
		rng: np.random.Generator,
	) -> np.ndarray:
		"""Calcule les moyennes des clusters et gère les clusters vides."""
		n_features = x.shape[1]
		centroids = np.empty((self.n_clusters, n_features), dtype=float)

		for cluster_idx in range(self.n_clusters):
			cluster_points = x[labels == cluster_idx]

			if cluster_points.size == 0:
				# Cas cluster vide : on ré-ensemence le centroïde avec le point
				# le plus éloigné de son centroïde actuel.
				farthest_idx = self._index_of_farthest_point(
					x,
					labels,
					previous_centroids,
				)
				centroids[cluster_idx] = x[farthest_idx]

				# Un petit bruit aléatoire évite les centroïdes dupliqués.
				centroids[cluster_idx] += rng.normal(0.0, 1e-6, size=n_features)
			else:
				centroids[cluster_idx] = cluster_points.mean(axis=0)

		return centroids

	@staticmethod
	def _index_of_farthest_point(
		x: np.ndarray,
		labels: np.ndarray,
		centroids: np.ndarray,
	) -> int:
		"""Trouve le point le plus éloigné de son centroïde assigné."""
		assigned_centroids = centroids[labels]
		distances = np.linalg.norm(x - assigned_centroids, axis=1)
		return int(np.argmax(distances))

	@staticmethod
	def _compute_inertia(
		x: np.ndarray,
		labels: np.ndarray,
		centroids: np.ndarray,
	) -> float:
		"""Inertie = somme des distances au carré au centroïde le plus proche."""
		squared_distances = np.sum((x - centroids[labels]) ** 2, axis=1)
		return float(np.sum(squared_distances))

	def _check_is_fitted(self) -> None:
		if self.cluster_centers_ is None:
			raise RuntimeError("Ce modèle KMeans n'est pas entraîné. Lance fit() d'abord.")

	def _validate_input(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=float)

		if x.ndim != 2:
			raise ValueError("x doit être un tableau 2D de forme (n_samples, n_features).")
		if x.shape[0] < self.n_clusters:
			raise ValueError("n_samples doit être >= n_clusters.")
		if x.shape[1] == 0:
			raise ValueError("x doit contenir au moins une feature.")

		return x


def _build_demo_data(seed: int = 7) -> np.ndarray:
	"""Construit un petit jeu de données synthétiques sans dépendance ML externe."""
	rng = np.random.default_rng(seed)

	cluster_a = rng.normal(loc=(-4.0, -2.0), scale=0.7, size=(90, 2))
	cluster_b = rng.normal(loc=(0.5, 3.0), scale=0.8, size=(110, 2))
	cluster_c = rng.normal(loc=(4.5, -1.0), scale=0.9, size=(100, 2))
	return np.vstack([cluster_a, cluster_b, cluster_c])


# ---------------------------------------------------------------------------
# Métriques internes de qualité d'un clustering (from scratch, NumPy only)
# ---------------------------------------------------------------------------


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
	"""Score de silhouette moyen pour un clustering donné.

	Pour chaque point i on calcule :
	    s(i) = (b(i) - a(i)) / max(a(i), b(i))
	où :
	    a(i) = distance moyenne de i aux autres points de **son** cluster
	    b(i) = distance moyenne minimale de i aux points d'un **autre** cluster

	Parameters
	----------
	x : np.ndarray de forme (n_samples, n_features)
		Les données.
	labels : np.ndarray de forme (n_samples,)
		Les étiquettes de cluster pour chaque point.

	Returns
	-------
	float
		Score de silhouette moyen (entre -1 et 1). Plus c'est proche de 1,
		meilleur est le clustering.
	"""
	x = np.asarray(x, dtype=float)
	labels = np.asarray(labels, dtype=int)
	unique_labels = np.unique(labels)
	n_clusters = len(unique_labels)

	if n_clusters <= 1:
		# Un seul cluster : le score n'a pas de sens, on retourne -1.
		return -1.0

	n_samples = x.shape[0]

	# Matrice de distances pré-calculée pour éviter des calculs redondants.
	# On ne garde que le triangle supérieur pour la mémoire, mais ici on
	# privilégie la lisibilité.
	distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)  # (n, n)

	silhouette_vals = np.empty(n_samples, dtype=float)

	for i in range(n_samples):
		cluster_i = labels[i]

		# Masques
		same_cluster_mask = labels == cluster_i
		same_cluster_mask[i] = False  # exclut le point lui-même

		# a(i) : distance moyenne intra-cluster
		if np.any(same_cluster_mask):
			a_i = distances[i, same_cluster_mask].mean()
		else:
			a_i = 0.0

		# b(i) : plus petite distance moyenne inter-cluster
		b_i = float("inf")
		for other_label in unique_labels:
			if other_label == cluster_i:
				continue
			other_mask = labels == other_label
			mean_dist = distances[i, other_mask].mean()
			if mean_dist < b_i:
				b_i = mean_dist

		if b_i == float("inf"):
			b_i = 0.0

		denom = max(a_i, b_i)
		silhouette_vals[i] = (b_i - a_i) / denom if denom > 0 else 0.0

	return float(silhouette_vals.mean())


def davies_bouldin_score(x: np.ndarray, labels: np.ndarray) -> float:
	"""Indice de Davies-Bouldin pour un clustering donné.

	Pour chaque cluster k on définit :
	    R_k = max_{j ≠ k} ( (s_k + s_j) / d(c_k, c_j) )
	où s_k est la dispersion moyenne intra-cluster et d(c_k, c_j) la distance
	entre les centroïdes des clusters k et j.

	Le score final est la moyenne des R_k sur tous les clusters.
	**Plus le score est petit, meilleur est le clustering.**

	Parameters
	----------
	x : np.ndarray de forme (n_samples, n_features)
		Les données.
	labels : np.ndarray de forme (n_samples,)
		Les étiquettes de cluster pour chaque point.

	Returns
	-------
	float
		Indice de Davies-Bouldin.
	"""
	x = np.asarray(x, dtype=float)
	labels = np.asarray(labels, dtype=int)
	unique_labels = np.unique(labels)
	n_clusters = len(unique_labels)

	if n_clusters <= 1:
		return float("inf")

	# Calcul des centroïdes et des dispersions intra-cluster
	centroids = np.empty((n_clusters, x.shape[1]), dtype=float)
	intra_dispersion = np.empty(n_clusters, dtype=float)

	for idx, label in enumerate(unique_labels):
		cluster_points = x[labels == label]
		centroids[idx] = cluster_points.mean(axis=0)
		# Dispersion = distance moyenne des points au centroïde
		if len(cluster_points) > 0:
			intra_dispersion[idx] = np.linalg.norm(
				cluster_points - centroids[idx], axis=1
			).mean()
		else:
			intra_dispersion[idx] = 0.0

	r_values = np.empty(n_clusters, dtype=float)

	for k in range(n_clusters):
		max_ratio = 0.0
		for j in range(n_clusters):
			if j == k:
				continue
			# Distance entre centroïdes k et j
			centroid_dist = np.linalg.norm(centroids[k] - centroids[j])
			if centroid_dist == 0:
				ratio = float("inf")
			else:
				ratio = (intra_dispersion[k] + intra_dispersion[j]) / centroid_dist
			if ratio > max_ratio:
				max_ratio = ratio
		r_values[k] = max_ratio

	return float(r_values.mean())


def calinski_harabasz_score(x: np.ndarray, labels: np.ndarray) -> float:
	"""Indice de Calinski-Harabasz (Variance Ratio Criterion) pour un clustering.

	CH = ( trace(B_k) / trace(W_k) ) * ( (n - k) / (k - 1) )

	où :
	    B_k = matrice de dispersion inter-cluster
	    W_k = matrice de dispersion intra-cluster
	    n   = nombre total d'échantillons
	    k   = nombre de clusters

	**Plus le score est grand, meilleur est le clustering.**

	Parameters
	----------
	x : np.ndarray de forme (n_samples, n_features)
		Les données.
	labels : np.ndarray de forme (n_samples,)
		Les étiquettes de cluster pour chaque point.

	Returns
	-------
	float
		Indice de Calinski-Harabasz.
	"""
	x = np.asarray(x, dtype=float)
	labels = np.asarray(labels, dtype=int)
	unique_labels = np.unique(labels)
	n_clusters = len(unique_labels)
	n_samples = x.shape[0]

	if n_clusters <= 1 or n_clusters >= n_samples:
		return 0.0

	# Centre global des données
	global_center = x.mean(axis=0)

	# trace(W_k) : dispersion intra-cluster
	trace_wk = 0.0
	# trace(B_k) : dispersion inter-cluster
	trace_bk = 0.0

	for label in unique_labels:
		cluster_points = x[labels == label]
		n_k = cluster_points.shape[0]
		if n_k == 0:
			continue

		cluster_center = cluster_points.mean(axis=0)

		# Contribution intra-cluster : somme des distances au carré au centroïde
		trace_wk += np.sum((cluster_points - cluster_center) ** 2)

		# Contribution inter-cluster : n_k * distance² entre centroïde et centre global
		trace_bk += n_k * np.sum((cluster_center - global_center) ** 2)

	if trace_wk == 0:
		return float("inf")

	ch = (trace_bk / trace_wk) * ((n_samples - n_clusters) / (n_clusters - 1))
	return float(ch)


def _find_elbow_point(inertias: list[float]) -> int:
	"""Détecte le point de coude sur une courbe d'inertie.

	Utilise la méthode du « triangle » : pour chaque point on calcule la
	distance à la ligne reliant le premier et le dernier point de la courbe.
	Le K qui maximise cette distance est le coude.

	Parameters
	----------
	inertias : list[float]
		Inerties pour K = 1, 2, ..., n (dans l'ordre croissant).

	Returns
	-------
	int
		L'indice (0-based) du coude dans la liste fournie.
		Correspond au K = indice + 1 qui est le coude détecté.
	"""
	n = len(inertias)
	if n <= 2:
		return 0

	inertias_arr = np.array(inertias, dtype=float)
	x_vals = np.arange(1, n + 1, dtype=float)

	# Ligne entre le premier point (x=1, y=inertias[0]) et le dernier (x=n, y=inertias[-1])
	# Distance d'un point (x_i, y_i) à cette ligne :
	#   | (x2 - x1)*(y1 - y_i) - (x1 - x_i)*(y2 - y1) | / sqrt((x2 - x1)² + (y2 - y1)²)
	x1, y1 = x_vals[0], inertias_arr[0]
	x2, y2 = x_vals[-1], inertias_arr[-1]

	denom = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	if denom == 0:
		return 0

	numerators = np.abs((x2 - x1) * (y1 - inertias_arr) - (x1 - x_vals) * (y2 - y1))
	distances = numerators / denom

	# Le coude est le point le plus éloigné de la ligne
	return int(np.argmax(distances))


def find_optimal_k(
	x: np.ndarray,
	max_k: int = 10,
	random_state: int | None = None,
	init: str = "k-means++",
	max_iter: int = 300,
	tol: float = 1e-4,
	scoring: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict:
	"""Cherche le nombre optimal de clusters K en testant K = 2 → max_k.

	Pour chaque K, un modèle KMeans est entraîné puis les métriques de qualité
	sont calculées :
	    - inertie (somme des distances au carré au centroïde)
	    - score de silhouette (de -1 à 1, plus grand = meilleur)
	    - indice de Davies-Bouldin (plus petit = meilleur)
	    - indice de Calinski-Harabasz (plus grand = meilleur)

	Le « meilleur » K est retourné selon chaque critère.  La méthode du coude
	est appliquée sur la courbe d'inertie.

	Parameters
	----------
	x : np.ndarray de forme (n_samples, n_features)
		Les données à clusteriser.
	max_k : int, default=10
		Nombre maximal de clusters à tester (K maximum).
	random_state : int | None, default=None
		Graine aléatoire pour la reproductibilité.
	init : str, default="k-means++"
		Stratégie d'initialisation ("random" ou "k-means++").
	max_iter : int, default=300
		Nombre maximal d'itérations par entraînement.
	tol : float, default=1e-4
		Seuil de convergence par entraînement.
	scoring : callable | None, default=None
		Fonction de scoring personnalisée (x, labels) -> float.  Si fournie,
		elle est également évaluée pour chaque K.

	Returns
	-------
	dict
		Dictionnaire contenant :
		- "ks" : liste des K testés
		- "inertias", "silhouette_scores", "davies_bouldin_scores",
		  "calinski_harabasz_scores" : listes des métriques pour chaque K
		- "best_k_elbow", "best_k_silhouette", "best_k_davies_bouldin",
		  "best_k_calinski_harabasz" : K optimal selon chaque critère
		- "custom_scores" (si scoring est fourni)
		- "best_k_custom" (si scoring est fourni)
	"""
	x = np.asarray(x, dtype=float)
	n_samples = x.shape[0]
	max_k = min(max_k, n_samples - 1, n_samples)
	if max_k < 2:
		raise ValueError("max_k doit être >= 2 et < n_samples.")

	ks = list(range(2, max_k + 1))

	inertias: list[float] = []
	silhouette_scores: list[float] = []
	davies_bouldin_scores: list[float] = []
	calinski_harabasz_scores: list[float] = []
	custom_scores: list[float] | None = [] if scoring is not None else None

	for k in ks:
		model = KMeans(
			n_clusters=k,
			max_iter=max_iter,
			tol=tol,
			random_state=random_state,
			init=init,
		)
		model.fit(x)

		inertias.append(model.inertia_)
		silhouette_scores.append(silhouette_score(x, model.labels_))
		davies_bouldin_scores.append(davies_bouldin_score(x, model.labels_))
		calinski_harabasz_scores.append(calinski_harabasz_score(x, model.labels_))

		if scoring is not None:
			custom_scores.append(scoring(x, model.labels_))

	# Détermination du « meilleur K » selon chaque critère
	# Pour l'inertie : on inclut K=1 pour que le coude ait du sens.
	# On recalcule l'inertie pour K=1 avec un seul centroïde global.
	global_center = x.mean(axis=0)
	inertia_k1 = float(np.sum((x - global_center) ** 2))
	all_inertias = [inertia_k1] + inertias
	elbow_idx = _find_elbow_point(all_inertias)
	# elbow_idx = 0 correspond à K=1, 1 → K=2, etc.
	best_k_elbow = elbow_idx + 1
	# Si le coude tombe sur K=1, on prend K=2 car K=1 n'est pas un clustering utile
	if best_k_elbow < 2:
		best_k_elbow = 2

	silhouette_arr = np.array(silhouette_scores)
	best_k_silhouette = ks[int(np.argmax(silhouette_arr))]

	db_arr = np.array(davies_bouldin_scores)
	best_k_davies_bouldin = ks[int(np.argmin(db_arr))]

	ch_arr = np.array(calinski_harabasz_scores)
	best_k_calinski_harabasz = ks[int(np.argmax(ch_arr))]

	result: dict = {
		"ks": ks,
		"inertias": inertias,
		"silhouette_scores": silhouette_scores,
		"davies_bouldin_scores": davies_bouldin_scores,
		"calinski_harabasz_scores": calinski_harabasz_scores,
		"best_k_elbow": best_k_elbow,
		"best_k_silhouette": best_k_silhouette,
		"best_k_davies_bouldin": best_k_davies_bouldin,
		"best_k_calinski_harabasz": best_k_calinski_harabasz,
	}

	if custom_scores is not None:
		result["custom_scores"] = custom_scores
		custom_arr = np.array(custom_scores)
		result["best_k_custom"] = ks[int(np.argmax(custom_arr))]

	return result


if __name__ == "__main__":
	# Petit exemple exécutable pour valider rapidement la classe.
	data = _build_demo_data(seed=7)

	model = KMeans(
		n_clusters=3,
		max_iter=200,
		tol=1e-4,
		random_state=7,
		init="k-means++",
	)
	model.fit(data)

	print("Centroïdes finaux:")
	print(np.round(model.cluster_centers_, 3))
	print(f"Inertie: {model.inertia_:.3f}")
	print(f"Itérations: {model.n_iter_}")

	# Démonstration de la sélection automatique du K optimal
	print("\n" + "=" * 60)
	print("Recherche du K optimal (2 → 8) ...")
	print("=" * 60)
	results = find_optimal_k(data, max_k=8, random_state=7)

	for k, inert, sil, db, ch in zip(
		results["ks"],
		results["inertias"],
		results["silhouette_scores"],
		results["davies_bouldin_scores"],
		results["calinski_harabasz_scores"],
	):
		print(
			f"K={k:>2d}  Inertie={inert:>10.2f}  Silhouette={sil:>7.4f}  "
			f"Davies-Bouldin={db:>7.4f}  Calinski-Harabasz={ch:>10.2f}"
		)

	print(f"\nK optimal (coude)            : {results['best_k_elbow']}")
	print(f"K optimal (silhouette)       : {results['best_k_silhouette']}")
	print(f"K optimal (Davies-Bouldin)   : {results['best_k_davies_bouldin']}")
	print(f"K optimal (Calinski-Harabasz): {results['best_k_calinski_harabasz']}")
