# -*- coding: utf-8 -*-
import collections
import numpy as np
import hdbscan
from sklearn.cluster import KMeans

def compute_cluster_statistics(cluster_labels):
    cluster_sizes = collections.Counter(cluster_labels)
    print(f"\nNumber of clusters: {len(cluster_sizes)}")
    for cid, size in cluster_sizes.items():
        print(f" - Cluster {cid}: {size} items")
    if cluster_sizes:
        avg_size = sum(cluster_sizes.values()) / len(cluster_sizes)
        print(f"Average items per cluster: {avg_size:.2f}")
        
# ------------------- Clustering Methods -------------------
def cluster_with_hdbscan(embeddings, min_cluster_size=2, metric="euclidean"):
    #print("------------ Clustering with HDBSCAN ------------")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    clusters = clusterer.fit_predict(embeddings)
    return clusters


def cluster_with_kmeans(embeddings, num_clusters=10):
    #print("------------ Clustering with KMeans ------------")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(embeddings)
    return clusters
'''        
def cluster_with_hdbscan(embeddings, min_cluster_size=2, metric="euclidean"):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    return clusterer.fit_predict(embeddings)
'''

def compute_cluster_centroids(embeddings, labels):
    centroids = {}
    for cluster_id in set(labels):
        cluster_points = embeddings[labels == cluster_id]
        centroids[cluster_id] = np.mean(cluster_points, axis=0)
    return centroids