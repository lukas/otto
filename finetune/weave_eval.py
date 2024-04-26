from dataclasses import dataclass
from pathlib import Path
import simple_parsing
import os
import wandb
import weave
import asyncio

import torch
import datasets
from pydantic import model_validator
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaTokenizerFast
from transformers.models.mistral import MistralForCausalLM

mistral_prompt = """[INST] You are AI that converts human request into api calls. 
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
USER_QUERY: {user} [/INST]{answer}"""

os.environ["WEAVE_PARALLELISM"] = "1"


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

def hf_to_weave(dataset: datasets.Dataset, num_samples: int = None) -> weave.Dataset:
    "Convert a Huggingface dataset to a Weave dataset"
    list_ds = dataset.to_list()[0:num_samples]
    return weave.Dataset(rows=list_ds, name='test-ds')

def model_type(model_path: Path):
    try:
        if list(model_path.glob("*adapter*")):
            return AutoPeftModelForCausalLM
    except:
        return AutoModelForCausalLM

def load_model_and_tokenizer(model_at_or_hub):
    "Load model and tokenizer from W&B or HF"
    try:
        from wandb import Api
        api = Api()
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

@weave.op()
def match(answer: str, model_output: dict ) -> dict:
    "a row -> {'user': 'Cheers!', 'answer': 'other()'}"
    return {
        "acc": answer.strip() == model_output["generated_text"].strip(),
        "acc_lousy": answer.strip().lower() == model_output["generated_text"].strip().lower()
        }

class MistralFT(weave.Model):
    model_id: str
    system_prompt: str
    temperature: float = 0.5
    max_new_tokens: int = 128
    model: PeftModelForCausalLM | MistralForCausalLM | AutoModelForCausalLM
    tokenizer: LlamaTokenizerFast

    @model_validator(mode='before')
    def load_model_and_tokenizer(cls, v):
        model_id = v["model_id"]
        if model_id is None:
            raise ValueError("model_id is required")
        model, tokenizer = load_model_and_tokenizer(model_id)
        v["model"] = model
        v["tokenizer"] = tokenizer
        return v

    @weave.op()
    def format_prompt(self, user: str) -> str:
        return self.system_prompt.format(user=user, answer="")

    @weave.op()
    def predict(self, user: str) -> str:
        prompt = self.format_prompt(user)
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            outputs = self.model.generate(
                tokenized_prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
            )
        generated_text = self.tokenizer.decode(outputs[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
        return {"generated_text": generated_text}


if __name__ == "__main__":

    @dataclass
    class Model:
        model_id: str = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0'
        # model_id: str = 'meta-llama/Llama-2-7b-hf'
        temperature: float = 0.7
        max_new_tokens: int = 128
        system_prompt: str = mistral_prompt

    @dataclass
    class Config:
        dataset_at: str = 'capecape/otto/split_dataset:v2'
        num_samples: int = None
        model: Model = Model()

    args = simple_parsing.parse(Config)

    weave.init("otto11")

    # grab the dataset
    dataset = load_ds_from_artifact(args.dataset_at)

    # convert to weave
    wds = hf_to_weave(dataset["test"], args.num_samples)

    weave_model = MistralFT(
        model_id=args.model.model_id,
        system_prompt=args.model.system_prompt,
        temperature=args.model.temperature,
        max_new_tokens=args.model.max_new_tokens,
    )

    # print("sanity check")
    # print(f" >input: {wds.rows[0]}")
    # outputs = weave_model.predict(wds.rows[0]["user"])

    eval = weave.Evaluation(dataset=wds, scorers=[match])
    asyncio.run(eval.evaluate(weave_model))

