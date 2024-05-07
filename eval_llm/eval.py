## Eval your HF models with this file and save results to Weave
# 
# You need a GPU to make this go fast, at least 16GB of memory
#
# To run Vanilla Llama2
#   python eval.py --model_id meta-llama/Llama-2-7b-hf --prompt_file prompts/llama2_vanilla.txt
#
# Mistral Instruct:
#   python eval.py --model_id mistralai/Mistral-7B-Instruct-v0.2 --prompt_file prompts/mistral.txt
#
# To run finetuned Mistral
#   python eval.py --model_id capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0 \
#      --prompt_file prompts/mistral_simple.txt
#

from dataclasses import dataclass
from pathlib import Path
import simple_parsing
import os
import wandb
import weave
import asyncio

import torch
from pydantic import model_validator
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaTokenizerFast
from transformers.models.mistral import MistralForCausalLM
from transformers.models.llama import LlamaForCausalLM

from utils import load_ds_from_artifact, load_model_and_tokenizer, hf_dataset_to_weave

os.environ["WEAVE_PARALLELISM"] = "1"

WANDB_PROJECT = "otto_bis"

@weave.op()
def match(answer: str, model_output: dict ) -> dict:
    "a row -> {'user': 'Cheers!', 'answer': 'other()'}"
    return {
        "acc": answer.strip() == model_output["generated_text"].strip(),
        "acc_lousy": answer.strip().lower() == model_output["generated_text"].strip().lower()
        }

class HFModel(weave.Model):
    model_id: str
    system_prompt: str
    temperature: float = 0.5
    max_new_tokens: int = 128
    model: PeftModelForCausalLM | MistralForCausalLM | LlamaForCausalLM
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
        "A simple function to apply the chat template to the prompt"
        return self.system_prompt.format(user=user, answer="")

    @weave.op()
    def predict(self, user: str) -> str:
        prompt = self.format_prompt(user)
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            outputs = self.model.generate(
                tokenized_prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature>0),
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
            )
        generated_text = self.tokenizer.decode(outputs[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
        return {"generated_text": generated_text}


if __name__ == "__main__":

    @dataclass
    class ModelConfig:
        model_id: str = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0'
        temperature: float = 0.7
        max_new_tokens: int = 128
        prompt_file: Path = Path("prompts/mistral_simple.txt")

    @dataclass
    class Config:
        dataset_at: str = 'capecape/otto/split_dataset:v2'
        num_samples: int = None
        model: ModelConfig = ModelConfig()

    args = simple_parsing.parse(Config)

    system_prompt = Path(args.model.prompt_file).read_text()

    weave.init(WANDB_PROJECT)

    # grab the dataset
    dataset = load_ds_from_artifact(args.dataset_at)

    # convert to weave
    wds = hf_dataset_to_weave(dataset["test"], args.num_samples)

    weave_model = HFModel(
        model_id=args.model.model_id,
        system_prompt=system_prompt,
        temperature=args.model.temperature,
        max_new_tokens=args.model.max_new_tokens,
    )

    print("sanity check")
    print(f" >input: {wds.rows[0]}")
    outputs = weave_model.predict(wds.rows[0]["user"])
    print(f" >output: {outputs}")

    eval = weave.Evaluation(dataset=wds, scorers=[match])
    asyncio.run(eval.evaluate(weave_model))

