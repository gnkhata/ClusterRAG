# -*- coding: utf-8 -*-
from transformers import AutoModel, AutoTokenizer
from rank_bm25 import BM25Okapi
from pylate import models, rank
import numpy as np
import torch
from clusters.clustering import cluster_with_hdbscan, cluster_with_kmeans, compute_cluster_centroids

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ============================================================
# Utility Functions
# ============================================================

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]



# ============================================================
# Embedding & Clustering Utilities
# ============================================================
def compute_contr_or_bge_embed(model, tokenizer, texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = mean_pooling(outputs.last_hidden_state, tokens['attention_mask'])
    return embeddings.cpu().numpy()

def compute_colbert_embeddings(colbert, documents, device, batch_size, is_query=False):
    return colbert.encode(documents, batch_size=batch_size, is_query=is_query, show_progress_bar=False,)

def compute_cluster_centroids_tokenwise(embeddings, clusters):
    centroids = {}
    for cid in np.unique(clusters):
        cluster_points = [emb for emb, lbl in zip(embeddings, clusters) if lbl == cid]
        max_len = max(e.shape[0] for e in cluster_points)
        dim = cluster_points[0].shape[1]
        padded = [np.vstack([e, np.zeros((max_len - e.shape[0], dim))]) for e in cluster_points]
        centroids[cid] = np.mean(np.stack(padded), axis=0)
    return centroids

# ============================================================
# Retrievers or Rankers
# ============================================================

###-----contriever or  bge retriever-----####
def retrieve_top_cluster_with_contr_or_bge(model, tokenizer, corpus, profile, query, cluster_method="hdbscan", num_clusters=10):
    profile_emb = compute_contr_or_bge_embed(model, tokenizer, corpus)
    query_emb = compute_contr_or_bge_embed(model, tokenizer, [query]).squeeze()

    if cluster_method == "hdbscan":
        clusters = cluster_with_hdbscan(profile_emb, min_cluster_size=2, metric="euclidean")
    elif cluster_method == "kmeans":
        clusters = cluster_with_kmeans(profile_emb, num_clusters=10)
    else:
        raise ValueError(f"Unsupported clustering method: {cluster_method}")
    centroids = compute_cluster_centroids(profile_emb, clusters)

    if not centroids:
        sims = [cosine_similarity(query_emb, emb) for emb in profile_emb]
        ranked = sorted(zip(profile, sims), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked]

    sims = {cid: cosine_similarity(query_emb, c) for cid, c in centroids.items()}
    best_cluster = max(sims, key=sims.get)
    cluster_items = [(p, emb) for p, emb, lbl in zip(profile, profile_emb, clusters) if lbl == best_cluster]
    ranked = sorted(cluster_items, key=lambda x: cosine_similarity(query_emb, x[1]), reverse=True)
    return [p for p, _ in ranked]

###-----colbert ranker-----####
def retrieve_top_cluster_with_colbert(colbert, corpus, ids, profile, query, batch_size=32, cluster_method="hdbscan", num_clusters=10):
    profile_embs = compute_colbert_embeddings(colbert, corpus, device, batch_size, is_query=False)
    query_emb = compute_colbert_embeddings(colbert, query, device, batch_size, is_query=True).squeeze()

    cluster_embs = np.stack([np.mean(d, axis=0) for d in profile_embs])
    
    if cluster_method == "hdbscan":
        clusters = cluster_with_hdbscan(cluster_embs, min_cluster_size=2, metric="euclidean")
    elif cluster_method == "kmeans":
        clusters = cluster_with_kmeans(cluster_embs, num_clusters=10)
    else:
        raise ValueError(f"Unsupported clustering method: {cluster_method}")
    
    centroids = compute_cluster_centroids_tokenwise(profile_embs, clusters)

    if not centroids:  # fallback
        sims = rank.rerank(
            documents_ids=[ids],
            queries_embeddings=query_emb,
            documents_embeddings=[profile_embs]
        )[0]
        score_map = {x['id']: x['score'] for x in sims}
        return sorted([p for p in profile if p['id'] in score_map],
                      key=lambda x: score_map[x['id']], reverse=True)

    centroids = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in centroids.items()}
    sims = rank.rerank(
        documents_ids=[list(centroids.keys())],
        queries_embeddings=query_emb,
        documents_embeddings=[list(centroids.values())]
    )[0]

    selected = []
    for c in sims:
        cid = c["id"]
        cluster_items = {p["id"]: e for p, e, lbl in zip(profile, profile_embs, clusters) if lbl == cid}
        ranked = rank.rerank(
            documents_ids=[list(cluster_items.keys())],
            queries_embeddings=query_emb,
            documents_embeddings=[list(cluster_items.values())]
        )[0]
        for item in ranked:
            selected.append({"id": item["id"], "score": item["score"]})

    score_map = {x['id']: x['score'] for x in selected}
    filtered = [p for p in profile if p['id'] in score_map]
    return sorted(filtered, key=lambda x: score_map[x['id']], reverse=True)

###-----bm25 ranker-----####
def retrieve_top_k_with_bm25(corpus, profile, query):
    bm25 = BM25Okapi([x.split() for x in corpus])
    return bm25.get_top_n(query.split(), profile, n=len(profile))


# ============================================================
# Proxy Methods
# ============================================================
def load_ranking_model(ranker_name):
    """
    Loads the appropriate model and tokenizer for the given ranker.
    Returns model, tokenizer, and a string for reference.
    """
    print(f"Using retriever: {ranker_name}")
    if ranker_name == "colbert":
        model_name = "lightonai/colbertv2.0"
        model = models.ColBERT(model_name_or_path=model_name).to(device)
        tokenizer = None
        model.eval()
    else:
        if ranker_name == "contriever":
            model_name = "facebook/contriever"
        elif ranker_name == "bge":
            model_name = "BAAI/bge-base-en-v1.5"
        else:
            return None, None, None
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    return model, tokenizer, model_name


def rank_profile(ranker, model, tokenizer, corpus, ids, profile, query, batch_size, cluster_method, num_clusters):
    """
    Dispatches ranking based on the specified ranker.
    """
    if ranker == "contriever" or ranker == "bge":
        return retrieve_top_cluster_with_contr_or_bge(model, tokenizer, corpus, profile, query, cluster_method)
    elif ranker == "bm25":
        return retrieve_top_k_with_bm25(corpus, profile, query)
    elif ranker == "recency":
        return sorted(profile, key=lambda x: tuple(map(int, str(x.get('date', '0-0-0')).split("-"))), reverse=True)
    elif ranker == "colbert":
        return retrieve_top_cluster_with_colbert(model, corpus, ids, profile, query, batch_size, cluster_method, num_clusters)
    else:
        raise ValueError(f"Unsupported ranker: {ranker}")
