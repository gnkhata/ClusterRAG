# -*- coding: utf-8 -*-
"""
Zero-shot / inference-only evaluation for LaMP tasks
"""
import argparse
import os
import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
)
from transformers.data.data_collator import (
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
)

from data.datasets import (
    GeneralSeq2SeqDataset,
    convert_to_hf_dataset,
    create_preprocessor,
    get_all_labels,
)
from prompts.prompts import create_prompt_generator
from metrics.classification_metrics import (
    create_metric_f1_accuracy,
    create_metric_mae_rmse,
)
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from utils.filter_warnings import filter_warnings

filter_warnings()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True,choices=["google/flan-t5-xxl", "Qwen/Qwen2-7B-Instruct"])
parser.add_argument("--task", required=True, choices=["LaMP_1", "LaMP_2", "LaMP_3", "LaMP_4", "LaMP_5", "LaMP_7"])
parser.add_argument("--ranker", required=True, choices=["colbert", "contriever", "bge", "bm25", "recency", "random"])
parser.add_argument("--cluster_method", required=True, choices=["hdbscan", "kmeans"])
parser.add_argument("--num_retrieved", type=int, required=True)
parser.add_argument("--mode", required=True, choices=["collab", "no_collab", "hybrid"])
parser.add_argument("--use_profile", action="store_true")
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--generation_max_length", type=int, default=128)
parser.add_argument("--generation_num_beams", type=int, default=4)
parser.add_argument("--per_device_batch_size", type=int, default=1)
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0")

if __name__ == "__main__":
    start = time.time()
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_DEVICES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_path = os.path.join("data", "LaMP_Time_Based", opts.task)
    in_path = os.path.join(base_path, opts.ranker, opts.mode)

    paths = {
        "dev_path": os.path.join(in_path, f"{opts.cluster_method}_dev_ranked.json"),
        "out_dir": os.path.join(in_path, "output"),
    }
    os.makedirs(paths["out_dir"], exist_ok=True)

    model_name = opts.model_name.lower()


    tokenizer = AutoTokenizer.from_pretrained(
        opts.model_name, cache_dir=opts.cache_dir
    )

    # ------------------------------------------------
    # Model loading (NO quantization)
    # ------------------------------------------------
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            opts.model_name,
            cache_dir=opts.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            opts.model_name,
            cache_dir=opts.cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    
    if "t5" in model_name:
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            max_length=opts.max_length,
        )
    else:
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    
    prompt_generator = (
        create_prompt_generator(opts.num_retrieved, opts.max_length, tokenizer)
        if opts.use_profile else None
    )

   
    if opts.task in ["LaMP_1", "LaMP_2"]:
        labels = get_all_labels(opts.task)
        eval_dataset = GeneralSeq2SeqDataset(paths["dev_path"], opts.use_profile, opts.task, opts.mode, prompt_generator,)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels, model_name=model_name,)

    elif opts.task == "LaMP_3":
        labels = get_all_labels(opts.task)
        eval_dataset = GeneralSeq2SeqDataset(paths["dev_path"],opts.use_profile, opts.task, opts.mode, prompt_generator,)
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels, model_name=model_name,)

    elif opts.task in ["LaMP_4", "LaMP_5", "LaMP_7"]:
        eval_dataset = GeneralSeq2SeqDataset(
            paths["dev_path"], opts.use_profile, opts.task, opts.mode, prompt_generator
        )
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)
    else:  # LaMP_4, LaMP_5, LaMP_7
        print(f"Unrecognized task: {opts.task}")
    
    eval_dataset = convert_to_hf_dataset(
        eval_dataset, cache_dir=opts.cache_dir
    ).map(
        create_preprocessor(
            tokenizer=tokenizer,
            max_length=opts.max_length,
            model_name=model_name,
        ),
        batched=True,
    )

    if "t5" in model_name:
        eval_args = Seq2SeqTrainingArguments(
            output_dir=paths["out_dir"],
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=opts.per_device_batch_size,
            predict_with_generate=True,
            generation_max_length=opts.generation_max_length,
            generation_num_beams=opts.generation_num_beams,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            data_collator=collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        eval_args = TrainingArguments(
            output_dir=paths["out_dir"],
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=opts.per_device_batch_size,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )

    print("\nRunning zero-shot evaluation...")
    results = trainer.predict(eval_dataset)

    print("\n=== Evaluation Results ===")
    for k, v in results.metrics.items():
        print(f"{k}: {v:.4f}")
        
    print()
    elapsed = time.time() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Total run time: {int(h)}h {int(m)}m {s:.2f}s")
