
import os
import subprocess
import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{24000}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["user", "answer"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def create_prompt_formats(sample):


    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    INSTRUCTION_KEY = "### User:"
    RESPONSE_KEY = "### Answer:"

    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY} {sample['user']}"
    response = f"{RESPONSE_KEY}\n{sample['answer']}"


    formatted_prompt = f"{blurb}\n{instruction}\n{response}\n"

    sample["text"] = formatted_prompt

    return sample

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

from transformers.integrations import WandbCallback
import wandb

class WandbLlamaCallback(WandbCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("Step end ")
        # print("Kwargs  ", kwargs)
        return super().on_step_end(args, state, control, **kwargs)
    

    def on_epoch_end(self, args, state, control, **kwargs):
        print("EKwargs  ", kwargs)

        tokenizer = kwargs['tokenizer']
        train_dataloader = kwargs['train_dataloader']
        print(train_dataloader)
        print(tokenizer)
        for data in train_dataloader:
            print(data)
            print(dir(train_dataloader))
            tokenizer.decode(train_dataloader['input_ids'], skip_special_tokens=True)
         

        wandb.log({}, commit=False)
        
        super().on_epoch_end(args, state, control, **kwargs)
        
        print("Kwargs   ", kwargs)
        print("State    ", state)



def train(model, tokenizer, dataset, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)


    os.environ["WANDB_PROJECT"] = "otto" # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all" # log your models


    ds = dataset['train'].train_test_split(test_size=0.3)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=2000,
            eval_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            report_to="wandb",
        ),
        # callbacks=[WandbLlamaCallback()],

        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()






def merge_and_save_model(checkpoint_dir, merged_dir, base_model_name):
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir, device_map="auto", torch_dtype=torch.bfloat16)
    merged_model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(merged_dir)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--checkpoint_dir", type=str, default="models/final_checkpoint")
    argparse.add_argument("--base_model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    argparse.add_argument("--merged_dir", type=str, default="models/final_merged_checkpoint")
    argparse.add_argument("--llama_path", type=str, default="../llama.cpp")
    argparse.add_argument("--gguf-filename", type=str, default="ggml-finetuned-model-q4_0.gguf")
    argparse.add_argument("--load-dataset", action='store_true')
    argparse.add_argument("--train-model", action='store_true')
    argparse.add_argument("--merge-model", action='store_true')
    argparse.add_argument("--convert-model", action='store_true')
    argparse.add_argument("--all", action='store_true')
    args = argparse.parse_args()

    llama_model_path = os.path.join(args.llama_path, "models")
    llama_model_filename = os.path.join(llama_model_path, "models/ggml-model-q4_0.gguf")

    if args.all:
        args.load_dataset = True
        args.train_model = True
        args.merge_model = True
        args.convert_model = True

    if args.load_dataset:
        print("Preprocessing dataset")

        bnb_config = create_bnb_config()
        model, tokenizer = load_model(args.base_model_name, bnb_config)

        dataset = load_dataset("json", data_files="dataset/training_data.json")
        max_length = get_max_length(model)

        seed = 42
        dataset = preprocess_dataset(tokenizer, max_length, seed, dataset)

    if args.train_model:
        print("Training on dataset of length", len(dataset['train']))
        train(model, tokenizer, dataset, args.checkpoint_dir)

    if args.merge_model:
        print(f"Merging model and saving to {args.merged_dir}")
        merge_and_save_model(args.checkpoint_dir, args.merged_dir, args.base_model_name)

    if args.convert_model:

        print(f"Converting to gguf format and saving to {llama_model_filename}")

        subprocess.run(["python", os.path.join(args.llama_path, "convert.py"), args.merged_dir],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT)

        subprocess.run([os.path.join(args.llama_path, "quantize"), os.path.join(args.merged_dir, "ggml-model-f16.gguf"), os.path.join(llama_model_path, args.gguf_filename), "q4_0"])










