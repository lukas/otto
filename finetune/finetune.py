from dataclasses import dataclass
from pathlib import Path

import wandb
import simple_parsing
import torch
import datasets
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from peft import LoraConfig
from datasets import load_from_disk
from trl import SFTTrainer


WANDB_ROJECT = "otto_bis"

@dataclass
class ScriptArgs:
    "The Arguments for training the model"
    dataset_artifact: str='capecape/otto/split_dataset:v2' # "the W&B artifact holding the dataset
    prompt_file: Path='mistral_simple.txt' # "the system prompt"
    model_id: str="mistralai/Mistral-7B-Instruct-v0.2" # "the model name"
    output_dir: str="training_output" # "the output directory"
    learning_rate: float=1.4e-5 # "the learning rate"
    batch_size: int=2 # "the batch size", 24GB -> 2, 40GB -> 4
    seq_length: int=512 # "Input sequence length"
    gradient_accumulation_steps: int=4 # "simulate larger batch sizes"
    lora_r: int=64 # "the rank of the matrix parameter of the LoRA adapters"
    lora_alpha: int=16 # "the alpha parameter of the LoRA adapters"
    lora_dropout: float=0.05 # "the dropout parameter of the LoRA adapters"
    epochs: int=1 # "the number of training steps"
    packing: bool=True # "pack the input data"

def load_ds_from_artifact(at_address: str, type: str = "dataset") -> datasets.Dataset:
    "Load the dataset from an Artifact"
    artifact = wandb.use_artifact(at_address, type=type)
    artifact_dir = artifact.download()
    return load_from_disk(artifact_dir)

if __name__ == "__main__":
    args = simple_parsing.parse(ScriptArgs)
    # create W&B run
    wandb.init(project=WANDB_ROJECT)

    # load the dataset to train from
    ds = load_ds_from_artifact(args.dataset_artifact)

    # load the system prompt
    system_prompt = Path(args.prompt_file).read_text()
    print(f"You are using the following system prompt:\n{system_prompt}\n")

    # apply prompt per row of dataset
    ds = ds.map(lambda row: {"text": system_prompt.format_map(row)})

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Define the LoraConfig
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # Define the training arguments
    hf_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        num_train_epochs=args.epochs,
        report_to="wandb"
    )

    # Let's Train
    print(f"Training {args.model_id} on {len(ds['train'])} examples")
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        args=hf_args,
        max_seq_length=args.seq_length,
        packing=args.packing,
        peft_config=peft_config,
        dataset_text_field="text",
    )

    dl = trainer.get_train_dataloader()
    b = next(iter(dl))
    print("============= Dataloader Debug =============================")
    print(b["input_ids"].shape)
    for sample in b["input_ids"]:
        print(trainer.tokenizer.decode(sample))
        print("==========================================")

    trainer.train()

    # save model with tokenizer
    trainer.save_model("finetuned_model")
    print("Saving model as artifact to wandb")
    model_artifact = wandb.Artifact(
        name = f"{wandb.run.id}-mistral", 
        type="model",
        description="Model trained on syntethic function calls for Otto",
        metadata={"finetuned_from":args.model_id})
    model_artifact.add_dir("finetuned_model")
    wandb.log_artifact(model_artifact)

    # finish the run
    wandb.finish()
