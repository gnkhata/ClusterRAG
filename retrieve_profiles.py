# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
import os
import time
import json
import pickle
import torch
import argparse
import tqdm
from models.retrievers import rank_profile, load_ranking_model
from utils.get_corpus import load_get_corpus_func, get_corpus_ids
from utils.get_query import load_get_query_func
from utils.filter_warnings import filter_warnings

filter_warnings()

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--mode", required=True, choices=["collab", "no_collab", "hybrid"])
parser.add_argument("--ranker", required=True, choices=["colbert", "contriever", "bge", "bm25", "recency", "random"])
parser.add_argument("--stage", required=True, choices=["dev", "train", "test"])
parser.add_argument("--cluster_method", required=True, choices=["hdbscan", "kmeans"], help="Choose clustering algorithm: 'hdbscan' or 'kmeans'")
parser.add_argument("--use_date", action='store_true')
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1")
parser.add_argument("--emb_type", default="mean")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--input_path", default="")
parser.add_argument("--out_path", default="")
parser.add_argument("--max_retrieved_sim_users", type=int, default=1)
parser.add_argument("--dataset", default=None)
parser.add_argument("--num_clusters", type=int, default=20, help="Number of clusters (used only if --cluster_method kmeans is selected)")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def retrieve_max_similar_users(user_sims, max_sim_per_user):
    for uid, sims in user_sims.items():
        if len(sims) > max_sim_per_user:
            user_sims[uid] = sims[:max_sim_per_user]
    return user_sims


def retrieve_no_collab_profile(dataset):
    """
    Retrieves ranked profiles for each user in non-collaborative mode.
    Modular and efficient version.
    """
    print(f"Retrieving profile in {opts.mode} mode...")
    
    model, tokenizer, model_name = load_ranking_model(opts.ranker)

    if model_name:
        print(f"Loaded retrieval model: {model_name}")
    else:
        print(f"No model loading required for ranker: {opts.ranker}")

    for counter, data in enumerate(tqdm.tqdm(dataset, desc="Ranking profiles", disable=True)):
        input_string = data['input']
        profile = data['profile']

        # Build corpus and query
        corpus = get_corpus_func(profile, use_date=opts.use_date)
        ids = get_corpus_ids(profile)
        query = get_query_func(input_string)

        # Rank the profile using chosen ranker
        ranked_profile = rank_profile(opts.ranker, model, tokenizer, corpus, ids, profile, query, opts.batch_size, opts.cluster_method, opts.num_clusters)

        # Store results
        data['profile'] = ranked_profile

    print(f"Completed ranking for {len(dataset)} samples.")
    return dataset

def retrieve_collab_or_hybrid_profile(dataset, user_vocab, user_sims, mode):
    """Handles both collaborative and hybrid retrieval."""
    print(f"Retrieving profile in {mode} mode...")
    # Load model/tokenizer once (avoid reloading per sample)
    model, tokenizer, model_name = load_ranking_model(opts.ranker)
    
    updated_dataset = []

    max_user_sims = retrieve_max_similar_users(user_sims, opts.max_retrieved_sim_users)

    for idx, data in enumerate(tqdm.tqdm(dataset, desc="Retrieving profiles")):
        user_id = str(data['user_id'])
        query = get_query_func(data['input'])
        sim_users = max_user_sims.get(user_id, [])

        # Aggregate profiles
        collab_profiles = []
        for sim in sim_users:
            uid = sim['user_id']
            if uid in user_vocab:
                collab_profiles.extend(user_vocab[uid]['profile'])

        if not collab_profiles:
            continue
        
        for p in collab_profiles:
            if "corpus" in p:
                del p["corpus"]
        
        user_profile = data['profile']
        if not user_profile:
            continue
        
        #profiles = collab_profiles + user_profile
        if mode == "collab":
            profiles = collab_profiles
        
        elif mode == "hybrid":
            profiles = user_profile + collab_profiles  
        
        corpus   = get_corpus_func(profiles, use_date=opts.use_date)
        ids      = get_corpus_ids(profiles)
        
        # Rank the profile using chosen ranker
        ranked_profile  = rank_profile(opts.ranker, model, tokenizer, corpus, ids, profiles, query, opts.batch_size, opts.cluster_method, opts.num_clusters)
        
        data['profile'] = ranked_profile
        updated_dataset.append(data)

    return updated_dataset

def main():
    base_path = os.path.join("data", "LaMP_Time_Based", opts.task)
    opts.input_path = os.path.join(base_path, opts.stage)
    opts.out_path = os.path.join(base_path, opts.ranker)
    mode_out = os.path.join(opts.out_path, opts.mode)
    os.makedirs(mode_out, exist_ok=True)

    paths = {
        "dataset": os.path.join(opts.input_path, f"{opts.stage}_merged.json"),
        "user_vocab": os.path.join(opts.out_path, "user_vocab.pkl"),
        "user_sims": os.path.join(opts.out_path, f"{opts.cluster_method}_user_cluster_sim.json"),
        "ranked": os.path.join(mode_out, f"{opts.cluster_method}_{opts.stage}_ranked.json"),    
    }

    # Load data
    with open(paths["dataset"]) as f:
        dataset = json.load(f) 
        
    if os.path.exists(paths["user_vocab"]):
        with open(paths["user_vocab"], "rb") as f:
            user_vocab = pickle.load(f)
    else:
        with open(os.path.join(base_path, r"user_vocab.pkl"), "rb") as f:
            user_vocab = pickle.load(f)
            
    if os.path.exists(paths["user_sims"]):
        with open(paths["user_sims"]) as f:
            user_sims = json.load(f)
    else:
        with open(os.path.join(base_path, r"colbert", f"{opts.cluster_method}_user_cluster_sim.json")) as f:
            user_sims = json.load(f)

    print(f"Loaded {len(dataset)} data, {len(user_vocab)} vocabs, {len(user_sims)} user sims")

    start = time.time()
    if opts.mode == "collab" or opts.mode == "hybrid":
        dataset = retrieve_collab_or_hybrid_profile(dataset, user_vocab, user_sims, opts.mode)
    elif opts.mode == "no_collab":
        dataset = retrieve_no_collab_profile(dataset)
    else:
        raise NotImplementedError("Only collab, no-collab, and hybrid modes accepted")

    # Save results
    json.dump(dataset, open(paths["ranked"], "w"))

    elapsed = time.time() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"\nRanking saved to {paths['ranked']}")
    print(f"Total time: {int(h)}h {int(m)}m {s:.2f}s")

if __name__ == "__main__":
    opts = parser.parse_args()
    get_corpus_func = load_get_corpus_func(opts.task)
    get_query_func = load_get_query_func(opts.task)
    main()