# -*- coding: utf-8 -*-
import warnings
import re

def filter_warnings():
    # --- Suppress specific warnings cleanly ---
    warnings.filterwarnings("ignore", message=".*max_length.*")
    warnings.filterwarnings("ignore", message=".*gather along dimension 0.*")
    warnings.filterwarnings("ignore", message=".*NCCL.*")
    warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")
    warnings.filterwarnings(
        "ignore",
        message=re.escape("The following device_map keys do not match any submodules in the model")
    )
    warnings.filterwarnings(
        "ignore",
        message=re.escape("torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly")
    )
    warnings.filterwarnings(
        "ignore",
        message=re.escape("No label_names provided for model class")
    )
    warnings.filterwarnings(
        "ignore",
        message=re.escape("huggingface_hub cache-system uses symlinks by default")
    )
    warnings.filterwarnings(
        "ignore",
        message=re.escape("Past key values is deprecated and will be removed in Transformers")
    )
    warnings.filterwarnings(
        "ignore",
        message=re.escape("NCCL")
    )
