# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.utils.checkpoint
import time
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    BitsAndBytesConfig
)
from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from metrics.classification_metrics import create_metric_f1_accuracy, create_metric_mae_rmse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset
from prompts.prompts import create_prompt_generator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from utils.filter_warnings import filter_warnings

filter_warnings()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, choices=["google/flan-t5-base", "google/flan-t5-xxl", "Qwen/Qwen2-7B-Instruct"])
parser.add_argument("--task", required=True, choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--ranker", required=True, choices=["colbert", "contriever", "bge", "bm25", "recency"])
parser.add_argument("--cluster_method", required=True, choices=["hdbscan", "kmeans"], help="Choose clustering algorithm: 'hdbscan' or 'kmeans'")
parser.add_argument("--num_retrieved", type=int, required=True)
parser.add_argument("--mode", required=True, choices=["collab", "no_collab", "hybrid"])
parser.add_argument("--use_profile", action="store_true")
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--generation_max_length", type=int, default=128)
parser.add_argument("--per_device_batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr_scheduler_type", default="linear")
parser.add_argument("--warmup_ratio", type=float, default=0.05)
parser.add_argument("--generation_num_beams", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1")
parser.add_argument("--test_data", default="")


if __name__ == "__main__":
    start = time.time()
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_DEVICES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_path = os.path.join("data", "LaMP_Time_Based", opts.task)
    in_path = os.path.join(base_path, opts.ranker, opts.mode)
    out_dir = os.path.join(in_path, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    '''
    paths = {
        "train_path": os.path.join(in_path, f"{opts.cluster_method}_first100_train_questions.json"),
        "dev_path": os.path.join(in_path, f"{opts.cluster_method}_first100_dev_questions.json"),
        "out_dir": out_dir,
    }
    '''
    paths = {
        "train_path": os.path.join(in_path, f"{opts.cluster_method}_train_ranked.json"),
        "dev_path": os.path.join(in_path, f"{opts.cluster_method}_dev_ranked.json"),
        "out_dir": out_dir,
    }
    
    model_name = opts.model_name.lower()
    print(f"Loading model: {opts.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(opts.model_name, cache_dir=opts.cache_dir)

    # Detect if model is large (requires quantization)
    quantize_large_model = any(x in model_name for x in ["xxl", "llama", "qwen"])
    if quantize_large_model:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("Using 4-bit quantization for large model.")
    else:
        bnb_config = None

    # Load model
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            opts.model_name,
            cache_dir=opts.cache_dir,
            quantization_config=bnb_config,
            device_map="auto" if quantize_large_model else None,
            low_cpu_mem_usage=True,
        )
    elif "llama" in model_name or "qwen" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            opts.model_name,
            cache_dir=opts.cache_dir,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        raise ValueError(f"Unsupported model type: {opts.model_name}")

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    # --------------------------
    # Apply LoRA for quantized models
    # --------------------------
    if quantize_large_model:
        torch.utils.checkpoint.use_reentrant = False
        print("Applying LoRA fine-tuning adapters for quantized model...")
        model = prepare_model_for_kbit_training(model)
    
        # Auto-select LoRA target modules depending on architecture
        if "t5" in model_name:
            target_modules = ["q", "v"]
            task_type = "SEQ_2_SEQ_LM"
        elif "llama" in model_name:
            target_modules = ["q_proj", "v_proj"]
            task_type = "CAUSAL_LM"
        elif "qwen" in model_name:
            target_modules = ["q_proj", "v_proj"]
            task_type = "CAUSAL_LM"
        else:
            target_modules = ["q", "v"]
            task_type = "CAUSAL_LM"
    
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=task_type,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.tie_weights()
        model.config.use_cache = False
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    if "t5" in model_name:
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=opts.max_length)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    prompt_generator = create_prompt_generator(opts.num_retrieved, opts.max_length, tokenizer) if opts.use_profile else None

    greater_is_better = True
    if opts.task == "LaMP_1":
        train_dataset, labels = GeneralSeq2SeqDataset(paths["train_path"], opts.use_profile, opts.task, opts.mode, prompt_generator), get_all_labels(opts.task)
        eval_dataset = GeneralSeq2SeqDataset(paths["dev_path"], opts.use_profile, opts.task, opts.mode, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels, model_name=model_name)
        best_metric = "accuracy"
    elif opts.task == "LaMP_2":
        train_dataset, labels = GeneralSeq2SeqDataset(paths["train_path"], opts.use_profile, opts.task, opts.mode, prompt_generator), get_all_labels(opts.task)
        eval_dataset = GeneralSeq2SeqDataset(paths["dev_path"], opts.use_profile, opts.task, opts.mode, prompt_generator)
        compute_metrics = create_metric_f1_accuracy(tokenizer=tokenizer, all_labels=labels, model_name=model_name)
        best_metric = "accuracy"
    elif opts.task == "LaMP_3":
        train_dataset, labels = GeneralSeq2SeqDataset(paths["train_path"], opts.use_profile, opts.task, opts.mode, prompt_generator), get_all_labels(opts.task)
        eval_dataset = GeneralSeq2SeqDataset(paths["dev_path"], opts.use_profile, opts.task, opts.mode, prompt_generator)
        compute_metrics = create_metric_mae_rmse(tokenizer=tokenizer, all_labels=labels, model_name=model_name)
        best_metric = "mae"
        greater_is_better = False
    elif opts.task in ["LaMP_4", "LaMP_5", "LaMP_7"]:
        train_dataset = GeneralSeq2SeqDataset(paths["train_path"], opts.use_profile, opts.task, opts.mode, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(paths["dev_path"], opts.use_profile, opts.task, opts.mode, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer=tokenizer)
        best_metric = "rouge-1"
    else:
        raise ValueError(f"Unsupported task: {opts.task}")

    # --------------------------
    # Convert to HF Datasets
    # --------------------------
    train_dataset = convert_to_hf_dataset(train_dataset, cache_dir=opts.cache_dir).map(
        create_preprocessor(tokenizer=tokenizer, max_length=opts.max_length,model_name=model_name), batched=True
    )
    eval_dataset = convert_to_hf_dataset(eval_dataset, cache_dir=opts.cache_dir).map(
        create_preprocessor(tokenizer=tokenizer, max_length=opts.max_length,model_name=model_name), batched=True
    )
    
    
    #print(f"Sample 1; {train_dataset[0]}")
    # --------------------------
    # Training Arguments
    # --------------------------
    base_args = dict(
        output_dir=paths["out_dir"],
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=1 if quantize_large_model else opts.per_device_batch_size,
        per_device_eval_batch_size=1 if quantize_large_model else opts.per_device_batch_size,
        gradient_accumulation_steps=16 if quantize_large_model else opts.gradient_accumulation_steps,
        learning_rate=opts.learning_rate,
        weight_decay=opts.weight_decay,
        num_train_epochs=opts.epochs,
        lr_scheduler_type=opts.lr_scheduler_type,
        warmup_ratio=opts.warmup_ratio,
        save_strategy="epoch",
        logging_steps=50,
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=greater_is_better,
        bf16=True if torch.cuda.is_available() else False,
    )

    if "t5" in model_name:
        training_args = Seq2SeqTrainingArguments(
            **base_args,
            generation_num_beams=opts.generation_num_beams,
            predict_with_generate=True,
            generation_max_length=opts.generation_max_length,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics
        )
        if opts.task in ["LaMP_1", "LaMP_2", "LaMP_3"]:
            trainer.label_names = labels
    else:
        training_args = TrainingArguments(**base_args)

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
    trainer.train()
    elapsed = time.time() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Total run time: {int(h)}h {int(m)}m {s:.2f}s")
