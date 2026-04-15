# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import time
import argparse
import os
import pickle
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from clusters.clustering import cluster_with_hdbscan, cluster_with_kmeans
from models.embedding_model import EmbeddingModel

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--ranker", required=True, choices=["colbert",  "bge"])
parser.add_argument("--cluster_method", required=True, choices=["hdbscan", "kmeans"], help="Choose clustering algorithm: 'hdbscan' or 'kmeans'")
parser.add_argument("--min_cluster_size", type=int, default=2)
parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters (used only if --cluster_method kmeans is selected)")
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--input_path", default="")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--emb_model_pooling", default="average")
parser.add_argument("--emb_model_normalize", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512)

@torch.no_grad()
def get_emb(emb_model, tokenizer, batch_size, device, corpus, max_length):
    """Generate embeddings for a list of texts."""
    batched_corpus = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
    all_embs = None
    for batch in tqdm(batched_corpus, desc="Embedding batches"):
        tokens_batch = tokenizer(batch,
                                 padding=True,
                                 truncation=True,
                                 max_length=max_length,
                                 return_tensors='pt').to(device)
        batch_emb = emb_model(**tokens_batch).cpu()
        all_embs = batch_emb if all_embs is None else torch.cat((all_embs, batch_emb), dim=0)
    return all_embs


def embed_user_profiles(user_vocab, emb_model, tokenizer, batch_size, device, max_length):
    all_titles = []
    title_index_map = []  # track (user_id, profile_idx) for each title

    # Gather all titles to embed
    for user_id, user_data in user_vocab.items():
        for prof_idx, item in enumerate(user_data["profile"]):
            title_index_map.append((user_id, prof_idx))
            all_titles.append(item["corpus"])

    print(f"Total vocab items to embed: {len(all_titles)}")

    # Compute embeddings
    corpus_embs = get_emb(emb_model, tokenizer, batch_size, device, all_titles, max_length)

    # Replace "corpus" with "embed" (tensor)
    for emb, (user_id, p_idx) in zip(corpus_embs, title_index_map):
        item = user_vocab[user_id]["profile"][p_idx]
        item_id = item.get("id")  # safely get ID before deleting
        item.clear()              # remove all existing keys
        item["id"] = item_id      # restore ID
        item["embed"] = emb  
        
    return user_vocab

def compute_user_to_cluster_similarities(user_cluster_map, user_embeddings_dict, output_json):
    print("------------ Computing intra-cluster similarities ------------")
    cluster_to_users = defaultdict(list)

    # Group users by cluster
    for user_id, cluster_id in user_cluster_map.items():
        '''
        if cluster_id == -1:
            continue  # skip noise
        '''
        cluster_to_users[cluster_id].append(user_id)

    cluster_sims = {}
    print()
    for cluster_id, users in cluster_to_users.items():
        print(f"Processing cluster {cluster_id} with {len(users)} users")

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


# ------------------- User-Level Clustering -------------------
def cluster_users(user_dict, min_cluster_size, metric="euclidean", method="hdbscan", num_clusters=10):
    user_ids = []
    user_embeddings = []

    # --- Aggregate embeddings for each user ---
    for user_entry in user_dict.values():
        user_id = user_entry['user_id']
        profile_embs = []

        for profile in user_entry['profile']:
            emb = profile['embed']
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

    # --- Run selected clustering algorithm ---
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

    num_clusters_found = len([c for c in cluster_counts if c != -1])  # exclude noise for HDBSCAN
    items_per_cluster = {c: n for c, n in cluster_counts.items() if c != -1}
    avg_items_per_cluster = np.mean(list(items_per_cluster.values())) if items_per_cluster else 0

    # --- Print summary ---
    print("------------ User-Level Clustering Summary ------------")
    print(f"Method used: {method.upper()}")
    print(f"Number of clusters (excluding noise): {num_clusters_found}")
    print(f"Number of users per cluster: {items_per_cluster}")
    print(f"Average number of users per cluster: {avg_items_per_cluster:.2f}")
    print(f"Noise users (cluster -1): {cluster_counts.get(-1, 0)}")

    # --- Collect user embeddings dict for later similarity computation ---
    user_embeddings_dict = {int(uid): emb for uid, emb in zip(user_ids, user_embeddings)}

    # --- Return results ---
    return {
        "user_cluster_map": user_cluster_map,
        "cluster_counts": dict(cluster_counts),
        "num_clusters": num_clusters_found,
        "avg_items_per_cluster": avg_items_per_cluster,
        "user_embeddings_dict": user_embeddings_dict
    }


if __name__ == "__main__":
    start_time = time.time()
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES
    opts.input_path = os.path.join(r"data", r"LaMP_Time_Based", opts.task)
    cluster_sim_addr = os.path.join(opts.input_path, opts.ranker, f"{opts.cluster_method}_user_cluster_sim.json")
    os.makedirs(os.path.join(opts.input_path, opts.ranker), exist_ok=True)

    for flag, value in opts.__dict__.items():
        print(f'{flag}: {value}')

    if opts.ranker == "colbert":
        emb_model_path = "lightonai/colbertv2.0"
    elif opts.ranker == "bge":
        emb_model_path = "BAAI/bge-base-en-v1.5"
    emb_model = EmbeddingModel(
        emb_model_path, opts.emb_model_pooling, opts.emb_model_normalize
    ).eval().to(opts.device)
    emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_path)

    with open(os.path.join(opts.input_path, "user_vocab.pkl"), 'rb') as file:
        user_vocab = pickle.load(file)
    print(f"Number of users: {len(user_vocab)}")

    print("Generating embeddings for user profiles...")
    user_embeds = embed_user_profiles(user_vocab, emb_model, emb_tokenizer,
                                     opts.batch_size, opts.device, opts.max_length)
    
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
    '''
    print("Example user embedding entry:")
    first_key = next(iter(user_embeds))
    print(user_embed[first_key]["profile"][0])
    '''
    end_time = time.time()
    elapsed = end_time - start_time

    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"User embedding generation time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")