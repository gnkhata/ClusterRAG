# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import os
import json 
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["LaMP_1","LaMP_2","LaMP_3","LaMP_4","LaMP_5","LaMP_7"])
parser.add_argument("--stage", required=True, choices=["dev", "train", "test"])
parser.add_argument("--input_path", default="")
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1")

def merge(dataset, labels):
    for inp in dataset:
        for label in labels['golds']:
            if label['id'] == inp['id']:
                output = label['output']
                break
        inp['output'] = output
    return dataset


if __name__ == "__main__":
    start_time = time.time()
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES
    print(f"Task: {opts.task}")
    opts.input_path = os.path.join(r"data", r"LaMP_Time_Based", opts.task, opts.stage) 
    
    input_data_addr  = os.path.join(opts.input_path, f"{opts.stage}_questions.json")
    input_label_addr = os.path.join(opts.input_path, f"{opts.stage}_outputs.json")
    merged_addr   = os.path.join(opts.input_path, f"{opts.stage}_merged.json")

    if os.path.exists(input_data_addr):
        with open(input_data_addr) as file:
            dataset = json.load(file)
    else:
        raise FileNotFoundError(f"Input data path not found: {input_data_addr}")
        
    print(f"Loaded {len(dataset)} {opts.stage} questions")
        
    if os.path.exists(input_label_addr):
        with open(input_label_addr) as file:
            labels = json.load(file)
    else:
        raise FileNotFoundError(f"Input label path not found: {input_label_addr}")

    print(f"Loaded {len(labels['golds'])} {opts.stage} labels")
    
    merged = merge(dataset, labels)
    #print(f"Sample: {merged[0]}")
    with open(merged_addr, "w") as resfile:
        json.dump(merged, resfile, indent=4)
    print(f"Merged data saved to {merged_addr}")
    
    end_time = time.time()
    elapsed = end_time - start_time

    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Merging time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


