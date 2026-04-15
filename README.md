# ClusterRAG (Anonymous Repository)

This repository contains the implementation of **ClusterRAG: Cluster-Based Collaborative Filtering for  Personalized Retrieval-Augmented Generation**, a collaborative and hybrid user profiling framework for personalized text generation. The work has been submitted for review. ClusterRAG enhances retrieval-augmented generation by clustering similar users and leveraging both individual and collaborative profiles during generation.

The code is evaluated on the **LaMP benchmark**, following the experimental setup described in the accompanying paper.

---

## Dataset: LaMP Benchmark

ClusterRAG is evaluated on the **LaMP (Language Model Personalization) dataset**.

**Official LaMP dataset repository:**  
https://lamp-benchmark.github.io/download

### Dataset Setup (Required)

1. Download the LaMP dataset from the official repository.
2. Paste the downloaded files into the corresponding subdirectories inside the `data/` folder of this repository.
3. Ensure the directory structure matches the expected LaMP format before running any scripts.

> **Important:** The code will not run correctly unless the LaMP dataset is downloaded and placed in the correct subdirectories under `data/`. The execution steps below use LaMP_1, which can be replaced by the name of the desired task.

---

## Environment Setup

- Python ≥ 3.8  
- PyTorch  
- HuggingFace Transformers  
- ColBERTv2 by PyLate  
- HDBSCAN  
- NumPy, pandas, scikit-learn  

Dependencies can be installed based on the imports used in the scripts.

---

## Execution Steps

All commands should be executed from the repository root.

### Step 1: Generate User Sets
```bash
python user_embed/get_user_set.py --task LaMP_1 --use_date
```

### Step 2: Generate User Embeddings
```bash
python user_embed/get_user_embed.py --task LaMP_1 --ranker colbert
```

### Step 3: Collaborative User Clustering
```bash
python collab_filter/collab_filter_users.py --task LaMP_1 --ranker colbert --cluster_method hdbscan
```

### Step 4: Merge Training Data
```bash
python data/merge_data_label.py --task LaMP_1 --stage train
```

### Step 5: Merge Development Data
```bash
python data/merge_data_label.py --task LaMP_1 --stage dev
```

### Step 6: Merge Test Data
```bash
python data/merge_data_label.py --task LaMP_1 --stage test
```

### Step 7: Retrieve Profiles (Training)
```bash
python retrieve_profiles.py --task LaMP_1 --mode collab --ranker colbert --stage train --use_date --cluster_method hdbscan
```

### Step 8: Retrieve Profiles (Development – Hybrid Mode)
```bash
python retrieve_profiles.py --task LaMP_1 --mode hybrid --ranker colbert --stage dev --use_date --cluster_method hdbscan
```

### Step 9: Retrieve Profiles (Test)
```bash
python retrieve_profiles.py --task LaMP_1 --mode collab --ranker colbert --stage test --use_date --cluster_method hdbscan
```

---

## Model Inference Options

The final step supports **either fine-tuned LLMs or zero-shot LLMs**.

### Option A: Fine-Tuned LLM (Training + Evaluation)
```bash
python train_and_evaluate_llm.py \
  --task LaMP_1 \
  --model_name google/flan-t5-base \
  --mode collab \
  --ranker colbert \
  --use_profile \
  --max_length 512 \
  --num_retrieved 1 \
  --epochs 30 \
  --cluster_method hdbscan
```

### Option B: Zero-Shot LLM Evaluation
```bash
python zero_shot.py \
  --task LaMP_1 \
  --model_name google/flan-t5-xxl \
  --mode collab \
  --ranker colbert \
  --use_profile \
  --max_length 512 \
  --num_retrieved 1 \
  --cluster_method hdbscan
```

---

## Notes

- This repository is intentionally anonymized for double-blind review.
- All hyperparameters follow the configuration reported in the paper.
- The `--mode` argument supports `user-only`, `collab`, and `hybrid` settings where applicable.

---

## Citation

If you use this code, please cite the accompanying paper (citation will be added after the review process).
