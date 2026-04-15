# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import time
import argparse
import os
import pickle
import json
import glob
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from clusters.clustering import cluster_with_hdbscan, cluster_with_kmeans

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--ranker", required=True, choices=["colbert", "bge"])
parser.add_argument("--cluster_method", required=True, choices=["hdbscan", "kmeans"],
                    help="Choose clustering algorithm: 'hdbscan' or 'kmeans'")
parser.add_argument("--emb_type", default="mean")
parser.add_argument("--input_path", default="")
parser.add_argument("--min_cluster_size", type=int, default=2)
parser.add_argument("--num_clusters", type=int, default=10,
                    help="Number of clusters (used only if --cluster_method kmeans is selected)")


# ------------------- Intra-cluster Similarity -------------------
def compute_user_to_cluster_similarities(user_cluster_map, user_embeddings_dict, output_json):
    print("------------ Computing intra-cluster similarities ------------")
    cluster_to_users = defaultdict(list)

    # Group users by cluster
    for user_id, cluster_id in user_cluster_map.items():
        cluster_to_users[cluster_id].append(user_id)

    cluster_sims = {}
    print()
    for cluster_id, users in cluster_to_users.items():
        print(f"Processing cluster {cluster_id} with {len(users)} users")

        # Extract embeddings
        user_embs = []
        for u in users:
            emb = user_embeddings_dict[u]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            user_embs.append(emb)
        user_embs = np.vstack(user_embs)

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(user_embs)

        for i, uid in enumerate(users):
            sims = []
            for j, uid2 in enumerate(users):
                if i == j:
                    continue
                sims.append({"user_id": int(uid2), "score": float(sim_matrix[i, j])})
            sims = sorted(sims, key=lambda x: x["score"], reverse=True)
            cluster_sims[uid] = sims

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(cluster_sims, f, indent=4)
    print()
    print(f"Intra-cluster similarities saved to: {output_json}")

def cluster_users(user_dict, min_cluster_size, metric="euclidean", method="hdbscan", num_clusters=10):
    user_ids = []
    user_embeddings = []

    # --- Aggregate embeddings for each user ---
    for user_entry in user_dict.values():
        user_id = user_entry["user_id"]
        profile_embs = []

        for profile in user_entry["profile"]:
            emb = profile["embed"]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            profile_embs.append(emb)

        if len(profile_embs) == 0:
            continue  # skip users without embeddings

        mean_emb = np.mean(profile_embs, axis=0)
        user_ids.append(user_id)
        user_embeddings.append(mean_emb)

    user_embeddings = np.vstack(user_embeddings)
    print(f"Total users considered: {len(user_embeddings)}")

    if method.lower() == "hdbscan":
        clusters = cluster_with_hdbscan(user_embeddings, min_cluster_size=min_cluster_size, metric=metric)
    elif method.lower() == "kmeans":
        clusters = cluster_with_kmeans(user_embeddings, num_clusters=num_clusters)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # --- Map user to cluster ---
    user_cluster_map = {int(uid): int(cid) for uid, cid in zip(user_ids, clusters)}

    # --- Compute stats ---
    cluster_counts = defaultdict(int)
    for cid in clusters:
        cluster_counts[int(cid)] += 1

    num_clusters_found = len([c for c in cluster_counts if c != -1])
    items_per_cluster = {c: n for c, n in cluster_counts.items() if c != -1}
    avg_items_per_cluster = np.mean(list(items_per_cluster.values())) if items_per_cluster else 0

    # --- Print summary ---
    print("------------ User-Level Clustering Summary ------------")
    print(f"Method used: {method.upper()}")
    print(f"Number of clusters (excluding noise): {num_clusters_found}")
    print(f"Number of users per cluster: {items_per_cluster}")
    print(f"Average number of users per cluster: {avg_items_per_cluster:.2f}")
    print(f"Noise users (cluster -1): {cluster_counts.get(-1, 0)}")

    # --- Collect embeddings for similarity computation ---
    user_embeddings_dict = {int(uid): emb for uid, emb in zip(user_ids, user_embeddings)}

    return {
        "user_cluster_map": user_cluster_map,
        "cluster_counts": dict(cluster_counts),
        "num_clusters": num_clusters_found,
        "avg_items_per_cluster": avg_items_per_cluster,
        "user_embeddings_dict": user_embeddings_dict,
    }

def load_user_embedding_chunks(embed_dir, ranker, emb_type):
    """
    Load all user embedding chunks (colbert_mean_part_*.pkl) into one dictionary.
    """
    pattern = os.path.join(embed_dir, f"{ranker}_{emb_type}_part_*.pkl")
    chunk_files = sorted(glob.glob(pattern))

    # Fallback if only a single combined file exists
    if not chunk_files:
        single_path = os.path.join(embed_dir, f"{ranker}_{emb_type}.pkl")
        if os.path.exists(single_path):
            print(f"Loading single embedding file: {single_path}")
            with open(single_path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"No embedding files found in {embed_dir}")

    # Load chunks incrementally
    print(f"Found {len(chunk_files)} embedding chunks. Loading and merging...")
    user_embeds = {}
    for idx, path in enumerate(chunk_files):
        with open(path, "rb") as f:
            chunk_data = pickle.load(f)
            user_embeds.update(chunk_data)
        print(f"  Loaded chunk {idx + 1}/{len(chunk_files)}: {path} ({len(chunk_data)} users)")

    print(f"Total merged users: {len(user_embeds)}")
    return user_embeds

if __name__ == "__main__":
    start_time = time.time()
    opts = parser.parse_args()
    opts.input_path = os.path.join("data", "LaMP_Time_Based", opts.task, opts.ranker)

    embed_dir = os.path.join(opts.input_path, "user_embed")
    cluster_sim_addr = os.path.join(opts.input_path, f"{opts.cluster_method}_user_cluster_sim.json")

    user_embeds = load_user_embedding_chunks(embed_dir, opts.ranker, opts.emb_type)

    # Perform clustering
    results = cluster_users(
        user_embeds,
        min_cluster_size=opts.min_cluster_size,
        metric="euclidean",
        method=opts.cluster_method,
        num_clusters=opts.num_clusters
    )

    # Compute intra-cluster similarities
    compute_user_to_cluster_similarities(
        results["user_cluster_map"],
        results["user_embeddings_dict"],
        cluster_sim_addr
    )

    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total clustering and similarity computation time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
