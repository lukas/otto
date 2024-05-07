import shutil, os, subprocess
from pathlib import Path
from types import SimpleNamespace

import wandb

from ft_utils import (
    load_model_from_artifact, 
    load_model_from_hf,
)

args = SimpleNamespace(
    MODEL_AT = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0',
    MODEL_ID = None,  #  "mistralai/Mistral-7B-Instruct-v0.1" # Evaluate a model from the HF Hub
    CONVERT_MODEL = True, # Export to GGUF format
    GGUF_FILENAME = "mistral_instruct_ft",
    LLAMA_CPP_PATH = "../llama.cpp/",
    LLAMA_CPP_QUANTIZATION = "q4_0",
    LOG_MERGED = False, #
)

wandb.init(project="otto", job_type="merge_and_convert")

model, tokenizer = load_model_from_artifact(args.MODEL_AT)

## Merge Model
merged_model = model.merge_and_unload()
model_name = (merged_model.name_or_path).split("/")[1]  # extract original model name
merged_dir = f"output/{model_name}/merged"

# save the merged model and tokenizer
merged_model.save_pretrained(merged_dir, 
                             max_shard_size="2GB",
                             safe_serialization=True)
tokenizer.save_pretrained(merged_dir, legacy_format=None)
print(f"Saving model to: {merged_dir}")

# Add tokenizer.model file... 😭
# the llama tokenizer, same for mistral and llamas
tok_model = Path("tokenizer.model")
shutil.copy(tok_model, Path(merged_dir)/tok_model)

if args.LOG_MERGED:
    # store everything on a W&B Artifact
    at = wandb.Artifact(
        name=f"{model_name}-merged",
        type="model",
        description="Finetuned model on Otto dataset",
    )
    at.add_dir(merged_dir)  # add full model weights
    wandb.log_artifact(at)

## Start Convertion for llama.cpp usage
print(f"Converting to gguf format and saving to {merged_dir}")
subprocess.run(["python", os.path.join(args.LLAMA_CPP_PATH, "convert.py"), merged_dir],
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT)


fname = f"{args.GGUF_FILENAME}_{args.LLAMA_CPP_QUANTIZATION}.gguf"
output_model = os.path.join(merged_dir, fname)

subprocess.run([os.path.join(args.LLAMA_CPP_PATH, "quantize"), os.path.join(
    merged_dir, "ggml-model-f16.gguf"), output_model, args.LLAMA_CPP_QUANTIZATION])

artifact = wandb.Artifact('gguf-model-finetuned', 
                          type='model',
                          description="A finetuned model on Otto ds to use in llama.cpp",
                          metadata={"model_name": model_name,
                                    "quantization": args.LLAMA_CPP_QUANTIZATION},
)
                                    
artifact.add_file(output_model)
wandb.log_artifact(artifact)
wandb.finish()
