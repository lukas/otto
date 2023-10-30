import wandb
from types import SimpleNamespace
from tqdm.auto import tqdm

import torch
from transformers import GenerationConfig, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from ft_utils import (
    load_model_from_artifact, 
    generate, 
    load_ds_from_artifact, 
    create_mistral_instruc_prompt, 
    create_llama_prompt,
    load_model_from_hf,
)

defaults = SimpleNamespace(
    MODEL_AT = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0',
    DATASET_AT = 'capecape/otto/split_dataset:v2',
    MODEL_ID = None,  #  "mistralai/Mistral-7B-Instruct-v0.1" # Evaluate a model from the HF Hub
)

def load_ds_and_model(defaults):
    ds = load_ds_from_artifact(defaults.DATASET_AT)
    if defaults.MODEL_ID is not None:
        print(f"Loading pre-trained model, not finetuned one: {defaults.MODEL_ID}")
        model, tokenizer = load_model_from_artifact(defaults.MODEL_ID)
    else:
        model, tokenizer = load_model_from_artifact(defaults.MODEL_AT)
    model = getattr(model, "model", model)  # maybe unwrap model
    if "mistral" in model.name_or_path.lower():
        create_prompt = create_mistral_instruc_prompt
    else:
        create_prompt = create_llama_prompt
    create_test_prompt = lambda row: {"text": create_prompt({"user":row["user"], "answer":""})}
    test_dataset = ds["test"].map(create_test_prompt)
    
    return (model, tokenizer), test_dataset



def create_predictions_table(model, tokenizer, test_dataset, max_new_tokens=256):
    gen_config = GenerationConfig.from_pretrained(
        model.name_or_path,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens)

    records_table = wandb.Table(columns=["prompt", "user", "answer", "generation"] + list(gen_config.to_dict().keys()))

    for example in tqdm(test_dataset, leave=False):
        prompt = example["text"]
        generation = generate(prompt=prompt, 
                               model=model, 
                               tokenizer=tokenizer,
                               gen_config=gen_config)
        records_table.add_data(prompt,
                               example["user"],
                               example["answer"],
                               generation, *list(gen_config.to_dict().values()))
    return records_table

if __name__ == "__main__":
    # initialize run
    wandb.init(project="otto", job_type="eval")

    (model, tokenizer), test_dataset = load_ds_and_model(defaults)

    table = create_predictions_table(model, tokenizer, test_dataset, 64)

    wandb.log({"eval_predictions":table})
    wandb.finish()