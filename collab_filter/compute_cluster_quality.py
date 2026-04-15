# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import time
import argparse
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from sklearn.metrics import silhouette_score
from clusters.clustering import cluster_with_hdbscan, cluster_with_kmeans
from models.embedding_model import EmbeddingModel


# ----------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True,
                    choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--ranker", required=True, choices=["colbert", "bge"])
parser.add_argument("--cluster_method", required=True,
                    choices=["hdbscan", "kmeans"])
parser.add_argument("--min_cluster_size", type=int, default=2)
parser.add_argument("--num_clusters", type=int, default=10)
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--emb_model_pooling", default="average")
parser.add_argument("--emb_model_normalize", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512)


# ----------------------------------------------------------------------
# Memory-safe embedding function
# ----------------------------------------------------------------------
@torch.no_grad()
def embed_user_profiles(user_vocab, emb_model, tokenizer,
                        batch_size, device, max_length):

    index_list, text_list = [], []

    for user_id, user_data in user_vocab.items():
        for prof_idx, item in enumerate(user_data["profile"]):
            index_list.append((user_id, prof_idx))
            text_list.append(item["corpus"])

    print(f"Total vocab items to embed: {len(text_list)}")

    for start in tqdm(range(0, len(text_list), batch_size), desc="Embedding"):
        end = min(start + batch_size, len(text_list))
        batch_texts = text_list[start:end]
        batch_indices = index_list[start:end]

        tokens = tokenizer(batch_texts,
                           padding=True,
                           truncation=True,
                           max_length=max_length,
                           return_tensors='pt').to(device)

        batch_embs = emb_model(**tokens).cpu()

        for emb, (uid, pid) in zip(batch_embs, batch_indices):
            item = user_vocab[uid]["profile"][pid]
            item_id = item.get("id")
            item.clear()
            item["id"] = item_id
            item["embed"] = emb

    return user_vocab


# ----------------------------------------------------------------------
# User-level clustering
# ----------------------------------------------------------------------
def cluster_users(user_dict, method="hdbscan",
                  min_cluster_size=2, num_clusters=10, metric="euclidean"):

    user_ids, user_embeddings = [], []

    for user_entry in user_dict.values():
        uid = user_entry["user_id"]
        profile_embs = []

        for p in user_entry["profile"]:
            emb = p["embed"]
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            profile_embs.append(emb)

        if not profile_embs:
            continue

        user_ids.append(uid)
        user_embeddings.append(np.mean(profile_embs, axis=0))

    user_embeddings = np.vstack(user_embeddings)
    print(f"Total users considered: {len(user_embeddings)}")

    if method == "hdbscan":
        labels = cluster_with_hdbscan(
            user_embeddings,
            min_cluster_size=min_cluster_size,
            metric=metric
        )
    else:
        labels = cluster_with_kmeans(
            user_embeddings,
            num_clusters=num_clusters
        )

    cluster_counts = defaultdict(int)
    for c in labels:
        cluster_counts[int(c)] += 1

    print("\n------------ Clustering Summary ------------")
    print(f"Method: {method.upper()}")
    print(f"Clusters (excluding noise): {len([c for c in cluster_counts if c != -1])}")
    print(f"Users per cluster: {dict(cluster_counts)}")
    print(f"Noise users (-1): {cluster_counts.get(-1, 0)}")

    return user_embeddings, labels


# ----------------------------------------------------------------------
# Silhouette score computation (separate function)
# ----------------------------------------------------------------------
def compute_silhouette(embeddings, labels, metric="euclidean"):
    labels = np.array(labels)
    valid = labels != -1  # remove noise points

    if len(set(labels[valid])) < 2:
        print("Silhouette score not defined (less than 2 clusters).")
        return None

    return silhouette_score(embeddings[valid], labels[valid], metric=metric)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    opts = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_DEVICES
    opts.input_path = os.path.join("data", "LaMP_Time_Based", opts.task)

    for k, v in vars(opts).items():
        print(f"{k}: {v}")

    if opts.ranker == "colbert":
        emb_model_path = "lightonai/colbertv2.0"
    else:
        emb_model_path = "BAAI/bge-base-en-v1.5"

    emb_model = EmbeddingModel(
        emb_model_path,
        opts.emb_model_pooling,
        opts.emb_model_normalize
    ).eval().to(opts.device)

    tokenizer = AutoTokenizer.from_pretrained(emb_model_path)

    with open(os.path.join(opts.input_path, "user_vocab.pkl"), "rb") as f:
        user_vocab = pickle.load(f)

    print(f"Number of users: {len(user_vocab)}")

    print("\nGenerating user embeddings...")
    user_vocab = embed_user_profiles(
        user_vocab,
        emb_model,
        tokenizer,
        opts.batch_size,
        opts.device,
        opts.max_length
    )

    embeddings, labels = cluster_users(
        user_vocab,
        method=opts.cluster_method,
        min_cluster_size=opts.min_cluster_size,
        num_clusters=opts.num_clusters
    )

    sil_score = compute_silhouette(embeddings, labels)

    print("\n------------ Cluster Quality ------------")
    if sil_score is not None:
        print(f"Silhouette Score: {sil_score:.4f}")
    else:
        print("Silhouette Score: N/A")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")
