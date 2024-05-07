import argparse
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
    create_mistral_instruct_prompt, 
    create_llama_prompt,
    create_llama_chat_prompt,
    create_custom_prompt,
    load_model_from_hf,
    read_file,
)

def validate_prompt_format(prompt):
    if not ("user" in prompt and "answer" in prompt):
        raise Exception("The prompt is not formatted correctly, you need to provide: user, answer")
    else:
        return create_custom_prompt(prompt)

def get_default_create_prompt(args):
    "This should come from the model artifact metadata"
    if "mistral" in args.MODEL_AT.lower():
        create_prompt = create_mistral_instruct_prompt
    elif "llama" in args.MODEL_AT.lower():
        if "chat" in args.MODEL_AT.lower():
            create_prompt = create_llama_chat_prompt
        else:
            create_prompt = create_llama_prompt
    else:
        raise Exception("Model not recognized")
    return create_prompt

defaults = SimpleNamespace(
    MODEL_AT = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0',
    DATASET_AT = 'capecape/otto/split_dataset:v2',
    MODEL_ID = None,
    PROMPT = None,
)

def parse_args(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL_AT", type=str, default=defaults.MODEL_AT)
    parser.add_argument("--DATASET_AT", type=str, default=defaults.DATASET_AT)
    parser.add_argument("--MODEL_ID", type=str, default=defaults.MODEL_ID)
    parser.add_argument("--PROMPT", type=str, default=defaults.PROMPT)
    parser.add_argument("--PROMPT_FILE", type=str, default=None)
    return parser.parse_args()

def load_test_ds(args):
    if args.PROMPT is not None: # custom prompt
        create_prompt = validate_prompt_format(args.PROMPT)
    else:
        create_prompt = get_default_create_prompt(args)

    # to parse the hf dataset
    create_test_prompt = lambda row: {"text": create_prompt({"user":row["user"], "answer":""})}
    
    ds = load_ds_from_artifact(args.DATASET_AT)
    test_dataset = ds["test"].map(create_test_prompt)
    return test_dataset

def load_model(args):
    if args.MODEL_ID is not None:  # load pre-trained model
        print(f"Loading pre-trained model, not finetuned one: {args.MODEL_ID}")
        model, tokenizer = load_model_from_hf(args.MODEL_ID)
    else:
        model = getattr(model, "model", model)  # maybe unwrap model
    
    return model, tokenizer

def create_predictions_table(model, tokenizer, test_dataset, max_new_tokens=64):
    gen_config = GenerationConfig.from_pretrained(
        model.name_or_path,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens)

    records_table = wandb.Table(columns=["prompt", "user", "answer", "generation"] + list(gen_config.to_dict().keys()))
    acc = 0.
    acc_lousy = 0.
    for example in tqdm(test_dataset, leave=False):
        prompt = example["text"]
        user_query = example["user"]
        label = example["answer"]
        generation = generate(prompt=prompt, 
                               model=model, 
                               tokenizer=tokenizer,
                               gen_config=gen_config)
        
        print(f"prompt: {prompt}")
        print(f"{user_query} -> {generation}")
        records_table.add_data(prompt,
                               user_query,
                               label,
                               generation.strip(), 
                               *list(gen_config.to_dict().values()))
        
        if generation.strip() == label: acc += 1
        if label.lower() in generation.lower(): acc_lousy +=1
    
    return records_table, acc/len(test_dataset), acc_lousy/len(test_dataset)

def evaluate(args):
    # initialize run
    wandb.init(project="otto", job_type="eval", config=args)

    test_dataset = load_test_ds(args)
    model, tokenizer = load_model(args)

    table, acc, acc_lousy = create_predictions_table(model, tokenizer, test_dataset, 64)

    wandb.log({"eval_predictions":table})
    wandb.run.summary["acc"] = acc
    wandb.run.summary["acc_lousy"] = acc_lousy
    wandb.finish()
    
if __name__ == "__main__":
    args = parse_args(defaults)
    if args.PROMPT_FILE is not None:
        print(f"Reading prompt from file: {args.PROMPT_FILE}")
        args.PROMPT = read_file(args.PROMPT_FILE)
    evaluate(args)