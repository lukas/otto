import wandb
import weave
import asyncio
import os

from .metrics import match

from weave import weaveflow

from types import SimpleNamespace
from tqdm.auto import tqdm
from openai import OpenAI

import torch
from transformers import GenerationConfig, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from ft_utils import generate, load_model_from_artifact, load_model_from_hf




# def create_predictions_table(model, tokenizer, test_dataset, max_new_tokens=64):
#     gen_config = GenerationConfig.from_pretrained(
#         model.name_or_path,
#         pad_token_id=tokenizer.eos_token_id,
#         max_new_tokens=max_new_tokens)

#     records_table = wandb.Table(columns=["prompt", "user", "answer", "generation"] + list(gen_config.to_dict().keys()))
#     acc = 0.
#     acc_lousy = 0.
#     for example in tqdm(test_dataset, leave=False):
#         prompt = example["text"]
#         user_query = example["user"]
#         label = example["answer"]
#         generation = generate(prompt=prompt, 
#                                model=model, 
#                                tokenizer=tokenizer,
#                                gen_config=gen_config)
        
#         print(f"{user_query} -> {generation}")
#         records_table.add_data(prompt,
#                                user_query,
#                                label,
#                                generation.strip(), 
#                                *list(gen_config.to_dict().values()))
        
#         if generation.strip() == label: acc += 1
#         if label.lower() in generation.lower(): acc_lousy +=1
    
#     return records_table, acc/len(test_dataset), acc_lousy/len(test_dataset)

def load_model(args):
    if args.MODEL_ID is not None:  # load pre-trained model
        print(f"Loading pre-trained model, not finetuned one: {args.MODEL_ID}")
        model, tokenizer = load_model_from_hf(args.MODEL_ID)
    else:
        model, tokenizer = load_model_from_artifact(args.MODEL_AT)
        model = getattr(model, "model", model)  # maybe unwrap model
    
    return model, tokenizer

@weave.type()
class GenText(weaveflow.Model):
    m: dict
    tokenizer: dict
    gen_config: dict

    @weave.op()
    async def predict(self, example: dict) -> dict:
        prompt = example["text"]
        user_query = example["user"]
        label = example["answer"]
        generation = generate(prompt=prompt, 
                               model=self.m, 
                               tokenizer=self.tokenizer,
                               gen_config=self.gen_config)
        
        return {'generated_text': generation}


def evaluate_model(dataset_name: str):
    max_new_tokens=64
    
    # hf_model, tokenizer = load_model(args)
    # model, tokenizer = load_model_from_artifact('capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0')
    model, tokenizer = load_model_from_artifact('llm-play/otto/gguf-model-finetuned:v0')

    gen_config = GenerationConfig.from_pretrained(
        model.name_or_path,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens)
    
    dataset = weave.ref(dataset_name).get()


    weave_model = GenText(model, tokenizer, gen_config)
    eval = weaveflow.Evaluation(dataset, scorers=[match])
    asyncio.run(eval.evaluate(weave_model))
    # table, acc, acc_lousy = create_predictions_table(model, tokenizer, test_dataset, 64)

