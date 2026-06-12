"""Implementation from scratch de K-means avec NumPy uniquement.

Objectif pédagogique : chaque étape de l'algorithme est explicite.
"""

from __future__ import annotations

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
