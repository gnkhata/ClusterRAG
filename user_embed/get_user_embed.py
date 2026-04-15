# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import time
import argparse
import os
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from models.embedding_model import EmbeddingModel


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--ranker", required=True, choices=["colbert", "bge"])
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--emb_model_pooling", default="average")
parser.add_argument("--emb_type", default="mean")
parser.add_argument("--emb_model_normalize", type=int, default=1)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--chunk_size", type=int, default=200, help="Number of users per saved file chunk")


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
    """Generate embeddings and attach them to user vocab entries."""
    all_titles = []
    title_index_map = []  # track (user_id, profile_idx) for each title

    # Gather all texts to embed
    for user_id, user_data in user_vocab.items():
        for prof_idx, item in enumerate(user_data["profile"]):
            title_index_map.append((user_id, prof_idx))
            all_titles.append(item["corpus"])

    print(f"Total vocab items to embed: {len(all_titles)}")

    corpus_embs = get_emb(emb_model, tokenizer, batch_size, device, all_titles, max_length)

    # Replace corpus with embeddings
    for emb, (user_id, p_idx) in zip(corpus_embs, title_index_map):
        item = user_vocab[user_id]["profile"][p_idx]
        item_id = item.get("id")  # store before clearing
        item.clear()
        item["id"] = item_id
        item["embed"] = emb

    return user_vocab


def save_in_chunks(user_embed, base_path, ranker, emb_type, chunk_size=200):
    """Save large user embeddings dictionary in smaller .pkl files."""
    os.makedirs(base_path, exist_ok=True)
    items = list(user_embed.items())
    total = len(items)
    print(f"Saving user embeddings in chunks of {chunk_size} users each...")

    for i in range(0, total, chunk_size):
        chunk = dict(items[i:i + chunk_size])
        part_path = os.path.join(
            base_path, f"{ranker}_{emb_type}_part_{i // chunk_size}.pkl"
        )
        with open(part_path, "wb") as f:
            pickle.dump(chunk, f)
        print(f"  Saved {len(chunk)} users to {part_path}")

    print(f"All chunks saved under: {base_path}")


if __name__ == "__main__":
    start_time = time.time()
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES

    opts.input_path = os.path.join("data", "LaMP_Time_Based", opts.task)
    opts.user_emb_dir = os.path.join("data", "LaMP_Time_Based", opts.task, opts.ranker, "user_embed")

    for flag, value in opts.__dict__.items():
        print(f"{flag}: {value}")

    # Select embedding model
    if opts.ranker == "colbert":
        emb_model_path = "lightonai/colbertv2.0"
    elif opts.ranker == "bge":
        emb_model_path = "BAAI/bge-base-en-v1.5"

    emb_model = EmbeddingModel(
        emb_model_path, opts.emb_model_pooling, opts.emb_model_normalize
    ).eval().to(opts.device)
    emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_path)

    # Load user vocab
    with open(os.path.join(opts.input_path, "user_vocab.pkl"), "rb") as file:
        user_vocab = pickle.load(file)
    print(f"Number of users: {len(user_vocab)}")

    print("Generating embeddings for user profiles...")
    user_embed = embed_user_profiles(user_vocab, emb_model, emb_tokenizer,
                                     opts.batch_size, opts.device, opts.max_length)

    save_in_chunks(user_embed, opts.user_emb_dir, opts.ranker, opts.emb_type, opts.chunk_size)

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"User embedding generation time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


