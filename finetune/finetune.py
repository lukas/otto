
from transformers.integrations import WandbCallback
import wandb
import os
import subprocess
import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback
from transformers.training_args import TrainingArguments
import transformers

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
INSTRUCTION_KEY = "### User: "
RESPONSE_KEY = "### Answer: "


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{24000}MB"

    if (model_name.startswith('bert')):
        print("In local mode, using bert-base-uncased model for speed")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            is_decoder=True
        )
    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # dispatch efficiently the model on the available ressources
            max_memory={i: max_memory for i in range(n_gpus)},

        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    if (model_name.startswith('bert')):
        tokenizer.pad_token = tokenizer.sep_token
    else:
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
    dataset = dataset.map(create_prompt_formats)

    _preprocessing_function = partial(

        preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        # remove_columns=["user", "answer"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(
        sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset


def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            break
    if not max_length:
        max_length = 1024
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


def create_prompt(user, answer, include_response=True):

    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}{user}"
    if include_response:
        response = f"{RESPONSE_KEY}{answer}"
    else:
        response = f"{RESPONSE_KEY}"

    formatted_prompt = f"{blurb}\n{instruction}\n{response}"
    return formatted_prompt


def create_prompt_formats(sample, include_response=True):
    formatted_prompt = create_prompt(

        sample['user'], sample['answer'], include_response=include_response)

    sample["text"] = formatted_prompt

    return sample


def find_all_linear_names(model):

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():

        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

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


class WandbLlamaCallback(TrainerCallback):

    # def on_step_end(
    #     self,
    #     args: TrainingArguments,
    #     state: TrainerState,
    #     control: TrainerControl,
    #     **kwargs,
    # ):

    #     return super().on_step_end(args, state, control, **kwargs)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("Evaluate")
        breakpoint()

    def on_epoch_begin(self, args, state, control, **kwargs):
        dataset = kwargs['train_dataloader'].dataset
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']
        test_and_log_raw_data(dataset=dataset, model=model,
                              tokenizer=tokenizer, name="train_results")

    def on_epoch_end(self, args, state, control, **kwargs):

        super().on_epoch_end(args, state, control, **kwargs)

        print("Kwargs   ", kwargs)
        print("State    ", state)


def prepare_model_for_training(model):
    model.gradient_checkpointing_enable()

    # put back for new models
    model = prepare_model_for_kbit_training(model)

    if (isinstance(model, transformers.models.bert.modeling_bert.BertLMHeadModel)):
        modules = ['query', 'key']
    else:
        modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT

    peft_config = create_peft_config(modules)
    peft_model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(peft_model)
    return peft_model


def train(model, tokenizer, dataset, output_dir, use_cuda, num_train_epochs=20, load_from_checkpoint=False, checkpoint_dir=None):
    os.environ["WANDB_PROJECT"] = "otto"  # log to your project
    os.environ["WANDB_LOG_MODEL"] = "all"  # log your models

    if use_cuda == False:
        model = model.to(device)

        training_args = TrainingArguments(
            do_eval=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            num_train_epochs=num_train_epochs,
            eval_steps=10,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir="outputs",
            no_cuda=True,
            report_to="wandb",

        )
    else:
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            num_train_epochs=num_train_epochs,
            eval_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            report_to="wandb",
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        callbacks=[WandbLlamaCallback()],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # re-enable for inference to speed up predictions for similar inputs
    model.config.use_cache = False

    # SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0

    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        if (load_from_checkpoint):
            train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
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


def load_model_from_checkpoint(checkpoint_dir, base_model_name):
    if (base_model_name.startswith('bert')):
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_dir,
        )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_dir, device_map="auto", torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if (base_model_name.startswith('bert')):
        tokenizer.pad_token = tokenizer.sep_token
    else:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def merge_and_save_model(checkpoint_dir, merged_dir, base_model_name):

    model, tokenizer = load_model_from_checkpoint(
        checkpoint_dir, base_model_name)
    merged_model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)

    # save tokenizer for easy inference

    tokenizer.save_pretrained(merged_dir)


def get_output_text(model, tokenizer, prompt):

    inputs = tokenizer(prompt, padding=True, truncation=True, max_length=get_max_length(
        model), return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=100)
    output_text = tokenizer.decode(
        outputs[0], skip_special_tokens=True)

    return output_text


def test_and_log_raw_data(model, tokenizer, dataset, name, num_examples=10):
    raw_data = []
    for i, input_data in enumerate(dataset["input_ids"]):
        if i >= num_examples:
            break

        text = tokenizer.decode(input_data)
        prompt_no_response_key, correct_response = text.split(RESPONSE_KEY, 1)
        user_query = prompt_no_response_key.split(INSTRUCTION_KEY, 1)[1]

        prompt = prompt_no_response_key + RESPONSE_KEY
        inputs = tokenizer(prompt, padding=True, truncation=True, max_length=get_max_length(
            model), return_tensors='pt').to(device)
        raw_outputs = model.generate(**inputs, max_new_tokens=100)
        _, llm_response = tokenizer.decode(
            raw_outputs[0]).split(RESPONSE_KEY, 1)
        raw_data.append([prompt, user_query, correct_response, llm_response])

    wandb.log({name: wandb.Table(data=raw_data, columns=[
              "prompt", "query", "response", "output_text"])})


def test_model(model, tokenizer, dataset):
    test_and_log_raw_data(model, tokenizer, dataset['train'], "train_results")
    test_and_log_raw_data(model, tokenizer, dataset['test'], "test_results")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()

    argparse.add_argument("--checkpoint_dir", type=str,
                          default="models/final_checkpoint")
    argparse.add_argument("--base-model-name", type=str,
                          default="meta-llama/Llama-2-7b-hf")
    argparse.add_argument("--merged-dir", type=str,
                          default="models/final_merged_checkpoint")
    argparse.add_argument("--llama-path", type=str, default="../llama.cpp")
    argparse.add_argument("--gguf-filename", type=str,
                          default="ggml-finetuned-model-q4_0.gguf")
    argparse.add_argument("--training-data", type=str,
                          default="dataset/training_data.json")
    argparse.add_argument("--test-model", action='store_true')
    argparse.add_argument("--train-model", action='store_true')
    argparse.add_argument("--merge-model", action='store_true')
    argparse.add_argument("--convert-model", action='store_true',
                          help="Convert model to gguf format to run in llama.cpp")
    argparse.add_argument("--num-train-epochs", type=int, default=20)
    argparse.add_argument("--load-from-checkpoint", action='store_true')
    argparse.add_argument("--no-cuda", action='store_true',
                          help="Don't use cuda")
    argparse.add_argument("--bert", action='store_true',
                          help="Use bert for fast testing")
    argparse.add_argument("--wandb", action='store_true',
                          help="Log with wandb")

    argparse.add_argument("--all", action='store_true')

    args = argparse.parse_args()

    llama_model_path = os.path.join(args.llama_path, "models")
    llama_model_filename = os.path.join(
        llama_model_path, "models/ggml-model-q4_0.gguf")

    if args.wandb:
        wandb.init(project="otto")

    if args.bert:
        args.base_model_name = "bert-base-uncased"

    if args.no_cuda:
        use_cuda = False
        device = None
    else:
        use_cuda = True
        device = "cuda:0"

    if args.all:
        args.load_dataset = True
        args.train_model = True
        args.merge_model = True
        args.convert_model = True

    print("Loading model")
    if args.load_from_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        # model, tokenizer = load_model_from_checkpoint(args.checkpoint_dir, args.base_model_name)
    else:
        bnb_config = create_bnb_config()
        model, tokenizer = load_model(
            args.base_model_name, bnb_config)

    print("Loading dataset")
    dataset = load_dataset("json", data_files=args.training_data)
    max_length = 512  # get_max_length(model)

    seed = 42
    processed_dataset = preprocess_dataset(
        tokenizer, max_length, seed, dataset)

    split_dataset = processed_dataset['train'].train_test_split(test_size=0.3)

    if args.test_model:
        test_model(model, tokenizer, split_dataset)

    if args.train_model:
        print("Training on dataset of length", len(processed_dataset["train"]))
        if (not args.load_from_checkpoint):
            peft_model = prepare_model_for_training(model)
        else:
            peft_model = "meta-llama/Llama-2-7b-hf"

        train(peft_model, tokenizer, split_dataset,
              args.checkpoint_dir, use_cuda, args.num_train_epochs, args.load_from_checkpoint, args.checkpoint_dir)

    if args.merge_model:
        print(f"Merging model and saving to {args.merged_dir}")
        merge_and_save_model(args.checkpoint_dir,
                             args.merged_dir, args.base_model_name)

    if args.convert_model:

        # if (args.wandb):
        #    wandb.init(project="otto", job_type="convert-gguf")

        print(
            f"Converting to gguf format and saving to {llama_model_filename}")
        
        print(" ".join(["python", os.path.join(args.llama_path, "convert.py"), args.merged_dir]))

        subprocess.run(["python", os.path.join(args.llama_path, "convert.py"), args.merged_dir],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)

        print(" ".join([os.path.join(args.llama_path, "quantize"), os.path.join(
            args.merged_dir, "ggml-model-f16.gguf"), os.path.join(llama_model_path, args.gguf_filename), "q4_0"]))
        subprocess.run([os.path.join(args.llama_path, "quantize"), os.path.join(
            args.merged_dir, "ggml-model-f16.gguf"), os.path.join(llama_model_path, args.gguf_filename), "q4_0"])

        if (args.wandb):
            artifact = wandb.Artifact('gguf-model-finetuned', type='model')
            artifact.add_file(os.path.join(
                llama_model_path, args.gguf_filename))
            wandb.log_artifact(artifact)

        print("Done!")
