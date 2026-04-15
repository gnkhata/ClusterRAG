# -*- coding: utf-8 -*-

import sys
sys.path.append(".")
import argparse
import json
import os
import pickle
from tqdm import tqdm
from utils.get_corpus import load_get_corpus_func
import time

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--use_date", action='store_true')
parser.add_argument("--cut_his_len", type=int, default=100)

if __name__ == "__main__":
    start_time = time.time()
    opts = parser.parse_args()
    print("task: {}".format(opts.task))

    get_corpus_func = load_get_corpus_func(opts.task)
 
    # Input path
    opts.input_path = os.path.join(r"data", r"LaMP_Time_Based", opts.task)
    '''
    if os.path.exists(opts.input_path):
        train_questions = json.load(open(os.path.join(opts.input_path, r"train/first100_train_questions.json"), "r"))
        dev_questions = json.load(open(os.path.join(opts.input_path, r"dev/first100_dev_questions.json"), "r"))
    '''
    if os.path.exists(opts.input_path):
        train_questions = json.load(open(os.path.join(opts.input_path, r"train/train_questions.json"), "r"))
        dev_questions = json.load(open(os.path.join(opts.input_path, r"dev/dev_questions.json"), "r"))
        #test_questions = json.load(open(os.path.join(opts.input_path, r"test/test_questions.json"), "r"))
    else:
        raise FileNotFoundError(f"Input path not found: {opts.input_path}")

    print(f"Loaded {len(train_questions)} train and {len(dev_questions)} development questions")

    # === Build User Vocabulary ===
    user_vocab = {}
    #all_questions = train_questions + dev_questions+test_questions
    all_questions = train_questions + dev_questions
    for q in tqdm(all_questions, desc="Building user vocab"):
        user_id = q["user_id"]

        # Sort profile chronologically
        profile = sorted(q["profile"], key=lambda x: tuple(map(int, str(x["date"]).split("-"))))

        # Build corpus (title + abstract (+ date))
        corpus = get_corpus_func(profile, use_date=opts.use_date)

        # Replace title + abstract with combined corpus text
        for i, p in enumerate(profile):
            p["user_id"] = user_id
            p["corpus"] = corpus[i]   # add the corpus text

        # If user not yet added, initialize
        if user_id not in user_vocab:
            user_vocab[user_id] = {"user_id": user_id, "profile": profile}
        else:
            # Merge unseen documents if new profile differs
            existing_ids = {item["id"] for item in user_vocab[user_id]["profile"]}
            for p in profile:
                if p["id"] not in existing_ids:
                    user_vocab[user_id]["profile"].append(p)

    print(f"Number of unique users: {len(user_vocab)}")

    # === Statistics ===
    profile_lengths = [len(user_vocab[uid]["profile"]) for uid in user_vocab]
    if profile_lengths:
        avg_len = sum(profile_lengths) / len(profile_lengths)
        print(f"Average profile length: {avg_len:.2f}")
        print(f"Min: {min(profile_lengths)}, Max: {max(profile_lengths)}")

    with open(os.path.join(opts.input_path, "user_vocab.pkl"), "wb") as f:
        pickle.dump(user_vocab, f)

    print(f"User vocabulary saved to {os.path.join(opts.input_path, 'user_vocab.pkl')}")
    
    end_time = time.time()
    elapsed = end_time - start_time

    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"User set creation time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")