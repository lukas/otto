from pathlib import Path
from functools import partial

import wandb
import weave

from tqdm.auto import tqdm
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, GenerationConfig, Trainer, AutoTokenizer
from transformers.integrations import WandbCallback


def read_file(fname):
    "Read a file to string"
    with open(fname, "r") as f:
        file_content = f.read()
    return file_content

llama_prompt = read_file("prompts/llama2_vanilla.txt")
llama_chat_prompt = read_file("prompts/llama2_chat.txt")
mistral_prompt = read_file("prompts/mistral.txt")

def create_custom_prompt(prompt_template):
    def _inner(row):
        return prompt_template.format(**row)
    return _inner

create_llama_prompt = create_custom_prompt(llama_prompt)
create_llama_chat_prompt = create_custom_prompt(llama_chat_prompt)
create_mistral_instruct_prompt = create_custom_prompt(mistral_prompt)


def load_ds_from_artifact(at_address, type="dataset"):
    "Load the dataset from an Artifact"
    if wandb.run is not None:
        artifact = wandb.use_artifact(at_address, type=type)
    else:
        from wandb import Api
        api = Api()
        artifact = api.artifact(at_address, type=type)
    artifact_dir = artifact.download()
    return load_from_disk(artifact_dir)

def model_type(model_path):
    if list(model_path.glob("*adapter*")):
        from peft import AutoPeftModelForCausalLM
        return AutoPeftModelForCausalLM
    return AutoModelForCausalLM



def load_model_from_artifact(MODEL_AT):
    "Load model and tokenizer from W&B"
    if not wandb.run:
        from wandb import Api
        api = Api()
        artifact = api.artifact(MODEL_AT, type="model")
    else:
        artifact = wandb.use_artifact(MODEL_AT, type="model")
    artifact_dir = Path(artifact.download())
    
    model = model_type(artifact_dir).from_pretrained(
            artifact_dir, device_map="auto", torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_model_from_hf(model_id):
    "Load model and tokenizer from HF"
    
    model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def has_exisiting_wandb_callback(trainer: Trainer):
    for item in trainer.callback_handler.callbacks:
        if isinstance(item, WandbCallback):
            return True
    return False

@weave.op()
def generate(prompt:str, model:AutoModelForCausalLM, tokenizer:AutoTokenizer, gen_config:GenerationConfig) -> str:
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(inputs=tokenized_prompt, 
                                pad_token_id=tokenizer.eos_token_id,
                                generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256):
        super().__init__()
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
        self.generate = partial(generate, 
                                model=trainer.model, 
                                tokenizer=trainer.tokenizer, 
                                gen_config=self.gen_config)
        
        #  we need to know if a wandb callback already exists
        if has_exisiting_wandb_callback(trainer):
            # if it does, we need to remove it
            trainer.callback_handler.pop_callback(WandbCallback)

    def log_generations_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt[-1000:])
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        self._wandb.log({"sample_predictions":records_table})
    
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self.log_generations_table(self.sample_dataset)
        
        

@weave.op()
def token_accuracy(eval_preds:tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    tp = (predictions.reshape(-1) == labels.reshape(-1)).astype(np.float)
    return {"eval_accuracy": tp.mean()}