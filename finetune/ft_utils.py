from pathlib import Path
from functools import partial

import wandb

from tqdm.auto import tqdm
import torch
import numpy as np
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, GenerationConfig, Trainer, AutoTokenizer
from transformers.integrations import WandbCallback

llama_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### User: {user}
### Answer: {answer}"""


llama_chat_prompt = """<s>[INST] <<SYS>>
You are AI that converts human request into api calls. 
You have a set of functions:
-news(topic="[topic]") asks for latest headlines about a topic.
-math(question="[question]") asks a math question in python format.
-notes(action="add|list", note="[note]") lets a user take simple notes.
-openai(prompt="[prompt]") asks openai a question.
-runapp(program="[program]") runs a program locally.
-story(description=[description]) lets a user ask for a story.
-timecheck(location="[location]") ask for the time at a location. If no location is given it's assumed to be the current location.
-timer(duration="[duration]") sets a timer for duration written out as a string.
-weather(location="[location]") ask for the weather at a location. If there's no location string the location is assumed to be where the user is.
-other() should be used when none of the other commands apply

Reply with the corresponding function call only, be brief.
<</SYS>>

Here is a user request, reply with the corresponding function call, be brief.
USER_QUERY: {user}[/INST]{answer}"""

mistral_prompt = """[INST]You are AI that converts human request into api calls. 
You have a set of functions:
-news(topic="[topic]") asks for latest headlines about a topic.
-math(question="[question]") asks a math question in python format.
-notes(action="add|list", note="[note]") lets a user take simple notes.
-openai(prompt="[prompt]") asks openai a question.
-runapp(program="[program]") runs a program locally.
-story(description=[description]) lets a user ask for a story.
-timecheck(location="[location]") ask for the time at a location. If no location is given it's assumed to be the current location.
-timer(duration="[duration]") sets a timer for duration written out as a string.
-weather(location="[location]") ask for the weather at a location. If there's no location string the location is assumed to be where the user is.
-other() should be used when none of the other commands apply

Here is a user request, reply with the corresponding function call, be brief.
USER_QUERY: {user}[/INST]{answer}"""


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

def generate(prompt, model, tokenizer, gen_config):
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
        
        

def token_accuracy(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    tp = (predictions.reshape(-1) == labels.reshape(-1)).astype(np.float)
    return {"eval_accuracy": tp.mean()}