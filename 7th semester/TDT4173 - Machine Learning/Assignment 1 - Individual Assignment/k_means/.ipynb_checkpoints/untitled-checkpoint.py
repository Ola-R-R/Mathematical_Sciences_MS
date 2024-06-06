import numpy as np
import pandas as pd
import math

class KMeans:
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations
        self.centroids = None
    
    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.iterations):
            labels = self.predict(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.array_equal(self.centroids, new_centroids):
                print(f"Iterations {_+1}\n")
                break
            
            self.centroids = new_centroids
            
        return labels, self.centroids
    
    def predict(self, X):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2), axis=1)
        return labels

# Some utility functions

def euclidean_silhouette(X, z):
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))

    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = np.linalg.norm(X[in_cluster_a][:, np.newaxis] - X[in_cluster_b], axis=2)
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    a = D[np.arange(len(X)), z]
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))

def euclidean_distortion(X, z):
    distortion = 0.0
    clusters = np.unique(z)

    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += np.sum(np.linalg.norm(Xc - mu, axis=1)**2)

    return distortion
