from pathlib import Path

import torch
import datasets
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

import weave
import wandb

def load_ds_from_artifact(at_address: str, type: str = "dataset") -> datasets.Dataset:
    "Load the dataset from an Artifact"
    if wandb.run is not None:
        artifact = wandb.use_artifact(at_address, type=type)
    else:
        api = wandb.Api()
        artifact = api.artifact(at_address, type=type)
    artifact_dir = artifact.download()
    return load_from_disk(artifact_dir)

def hf_dataset_to_weave(dataset: datasets.Dataset, num_samples: int = None) -> weave.Dataset:
    "Convert a Huggingface dataset to a Weave dataset"
    list_ds = dataset.to_list()[0:num_samples]
    return weave.Dataset(rows=list_ds, name='test-ds')

def model_type(model_path: Path):
    try:
        if list(model_path.glob("*adapter*")):
            return AutoPeftModelForCausalLM
    except:
        return AutoModelForCausalLM

def load_model_and_tokenizer(model_at_or_hub: str):
    "Load model and tokenizer from W&B or HF"
    try:
        api = wandb.Api()
        artifact = api.artifact(model_at_or_hub, type="model")
        artifact_dir = Path(artifact.download())
        print(f"Loaded model from {artifact_dir}")
    except:
        artifact_dir = model_at_or_hub
    
    model = model_type(artifact_dir).from_pretrained(
        artifact_dir, device_map="auto", torch_dtype=torch.bfloat16, use_cache=True)

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer