from dataclasses import dataclass
from pathlib import Path
import simple_parsing
import os
import weave
import asyncio

import torch
import datasets
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from ft_utils import load_ds_from_artifact, llama_prompt, mistral_prompt

# export WEAVE_PARALLELISM=1
os.environ["WEAVE_PARALLELISM"] = "1"

@dataclass
class Args:
    model_id: str = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0'
    # model_id: str = 'meta-llama/Llama-2-7b-hf'
    dataset_at: str = 'capecape/otto/split_dataset:v2'
    num_samples: int = None

def hf_to_weave(dataset: datasets.Dataset, num_samples: int = None) -> weave.Dataset:
    "Convert a Huggingface dataset to a Weave dataset"
    list_ds = dataset.to_list()[0:num_samples]
    return weave.Dataset(rows=list_ds, name='test-ds')

def model_type(model_path):
    if list(model_path.glob("*adapter*")):
        return AutoPeftModelForCausalLM
    return AutoModelForCausalLM

def maybe_load_from_at(model_at_or_hub):
    "Load model from W&B or HF"
    try:
        from wandb import Api
        api = Api()
        artifact = api.artifact(model_at_or_hub, type="model")
        artifact_dir = Path(artifact.download())
        print(f"Loaded model from {artifact_dir}")
    except:
        artifact_dir = model_at_or_hub
    
    model = model_type(artifact_dir).from_pretrained(
        artifact_dir, device_map="auto", torch_dtype=torch.bfloat16)

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

if __name__ == "__main__":
    weave.init("otto11")
    
    args = simple_parsing.parse(Args)

    # grab the dataset
    dataset = load_ds_from_artifact(args.dataset_at)

    # convert to weave
    wds = hf_to_weave(dataset["test"], args.num_samples)

    system_prompt = mistral_prompt

    model, tokenizer = maybe_load_from_at(args.model_id)

    class MistralFT(weave.Model):
        system_prompt: str
        temperature: float = 0.5
        max_new_tokens: int = 128

        @weave.op()
        def format_prompt(self, user: str) -> str:
            return self.system_prompt.format(user=user, answer="")

        @weave.op()
        def predict(self, user: str) -> str:
            prompt = self.format_prompt(self.system_prompt, user)
            tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
            with torch.inference_mode():
                outputs = model.generate(
                    tokenized_prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=self.temperature,
                )
            generated_text = tokenizer.decode(outputs[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
            return {"generated_text": generated_text}
    
    weave_model = MistralFT(
        system_prompt=get_prompt(args.model_id),
        temperature=0.5,
        max_new_tokens=128,
    )

    # print("sanity check")
    # print(f" >input: {wds.rows[0]}")
    # outputs = weave_model.predict(wds.rows[0]["user"])

    eval = weave.Evaluation(dataset=wds, scorers=[match])
    asyncio.run(eval.evaluate(weave_model))
