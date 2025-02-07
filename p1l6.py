import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# load dataset
file_path = "/Users/rahullenka/Downloads/kmeans - kmeans_blobs.csv" 
df = pd.read_csv(file_path)
# normalising the dataset
df_normalized = (df - df.min()) / (df.max() - df.min())
# k means clustering starts
def select_random_centers(data, k):
    """randomly select k centroids from data points"""
    return data.sample(n=k).values
def assign_clusters(data, centroids):
    """assign each data point to the nearest centroid"""
    distances = np.linalg.norm(data.values[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)
def update_centroids(data, labels, k):
    """find new centroids as mean of assigned points"""
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])
def k_means(data, k, max_iters=100, tol=1e-4):
    """perform k-means clustering"""
    centroids = select_random_centers(data, k)
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids
k_values = [2, 3]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, k in enumerate(k_values):
    labels, centroids = k_means(df_normalized, k)
    axes[i].scatter(df_normalized["x1"], df_normalized["x2"], c=labels, cmap='viridis', alpha=0.6)
    axes[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label="Centroids")
    axes[i].set_title(f'K-Means Clustering (k={k})')
    axes[i].set_xlabel('x1')
    axes[i].set_ylabel('x2')
    axes[i].legend()
plt.tight_layout()
plt.show()
