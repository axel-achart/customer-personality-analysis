# Implémentation "from scratch" de K-Means (algorithme de Lloyd), sans lib de clustering.
import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # tire k points au hasard parmi les données comme centroïdes de départ
    def _init_centroids_random(self, X, rng):
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    # initialisation k-means++ : choisit des centroïdes de départ éloignés les uns des autres
    def _init_centroids_kmeans_plus_plus(self, X, rng):
        n_samples = X.shape[0]
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)

        first_idx = rng.randint(n_samples)
        centroids[0] = X[first_idx]

        # distance au carré de chaque point au centroïde le plus proche déjà choisi
        closest_dist_sq = np.sum((X - centroids[0]) ** 2, axis=1)

        for i in range(1, self.n_clusters):
            # probabilité proportionnelle au carré de la distance -> favorise les points isolés
            total = closest_dist_sq.sum()
            if total == 0:
                probabilities = np.full(n_samples, 1.0 / n_samples)
            else:
                probabilities = closest_dist_sq / total

            next_idx = rng.choice(n_samples, p=probabilities)
            centroids[i] = X[next_idx]

            new_dist_sq = np.sum((X - centroids[i]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centroids

    def _init_centroids(self, X, rng):
        if self.init == "random":
            return self._init_centroids_random(X, rng)
        return self._init_centroids_kmeans_plus_plus(X, rng)

    # distances euclidiennes entre chaque point et chaque centroïde
    @staticmethod
    def _compute_distances(X, centroids):
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    # assigne chaque point au centroïde le plus proche
    def _assign_clusters(self, X, centroids):
        distances = self._compute_distances(X, centroids)
        return np.argmin(distances, axis=1), distances

    # recalcule chaque centroïde comme le barycentre de son cluster
    def _update_centroids(self, X, labels, centroids):
        new_centroids = centroids.copy()
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # cluster vide -> on le replace sur le point le plus isolé
                distances = self._compute_distances(X, new_centroids)
                farthest_idx = np.argmax(np.min(distances, axis=1))
                new_centroids[k] = X[farthest_idx]
        return new_centroids

    def _compute_inertia(self, X, labels, centroids):
        diff = X - centroids[labels]
        return float(np.sum(diff ** 2))

    # une exécution complète de l'algorithme (init -> boucle assign/update -> convergence)
    def _run_single(self, X, rng):
        centroids = self._init_centroids(X, rng)

        for iteration in range(1, self.max_iter + 1):
            labels, _ = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels, centroids)

            # convergence si les centroïdes ne bougent quasiment plus
            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)).max()
            centroids = new_centroids
            if shift <= self.tol:
                break

        labels, _ = self._assign_clusters(X, centroids)
        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration

    # API calquée sur sklearn : fit / predict / fit_predict
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_clusters ne peut pas être supérieur au nombre d'échantillons.")

        rng = np.random.RandomState(self.random_state)

        best_inertia = None
        best_centroids = None
        best_labels = None
        best_n_iter = None

        # plusieurs runs avec des initialisations différentes -> on garde la meilleure (inertie min)
        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._run_single(X, rng)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("Le modèle doit être entraîné avec fit() avant predict().")
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    # petit test rapide pour vérifier que l'implémentation tient la route,
    # en comparant avec la version scikit-learn sur le dataset Iris
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans as SklearnKMeans

    X_scaled = StandardScaler().fit_transform(load_iris().data)
    k = 3

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    print(f"Notre KMeans   : inertie = {model.inertia_:.2f} (convergence en {model.n_iter_} itérations)")
    print(f"Distribution des clusters : {np.bincount(labels)}")

    sk_model = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
    sk_model.fit(X_scaled)
    print(f"KMeans sklearn : inertie = {sk_model.inertia_:.2f}")