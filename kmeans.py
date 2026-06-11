import numpy as np
class kmeans():
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        np.random.seed(self.random_state)
        #initialize centroids randomly with random.choice
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        #assign points to nearest centroid
        for _ in range(self.max_iters):
            distances = self._calculate_distances(X)
            labels = np.argmin(distances, axis=0)
        #update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0)
                for k in range(self.n_clusters)
            ])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return labels

    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=0)


    def _calculate_distances(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((self.n_clusters, n_samples))
        for i in range(n_samples):
            diff = self.centroids - X[i]
            distances[:,i] = np.linalg.norm(diff, axis=1)
        return distances