# Fine Tuning Steps

We will fine-tune an OSS model adhere better at our instructions. This model's main purpose if formatting the instruction so we call the skill accordingly to the user request. We basically need a model that is capable of:
- parsing the input text from whisper
- deciding if the input text is a request for a skill
- deciding which skill is requested
- calling the skill with the right parameters, eg: "What is the weather in Berlin?" -> `weather(location="Berlin")`


## Install requirements

> We did our experimentation on a mixture of A100, A10s and 4090s. Check the [W&B workspace here](https://wandb.ai/l2k2/otto)

```bash
pip install -r requirements_finetune.txt
```

## Generate training data

To generate the training data we need to provide a list of examples for each skill. You can check the provided examples [here](examples/). If you want to add a new skill, you will need to retrain the model to obtain the best possible performance.

Each file is organized as following:
- Prompt: This section describes the skill, for instance for the `time` skill we have:
```
#### Prompt:
timecheck(location="[location]") ask for the time at a location. 
If no location is given it's assumed to be the current location.
```
then we have manually created examples of the prompts:
```
#### Prompt Examples:

### User: What time is it now?
### Assistant: timecheck()

### User: Whats the time
### Assistant: timecheck()
```
and finally we have the examples of the skill call that we can generate using OpenAI GPT that looks exactly like the prompt examples (if everything went well). I actually review them manually to make sure they are correct before appending them to the file.

We are using OpenAI function calls to restrict the output format to a specific one.

## Using GPT3.5-turbo/GPT-4 to generate examples

This script creates the example requests using GPT. It will print the generated examples to the console. You can then copy and paste them into the example file. It generates 5 examples at a time, you can tweak this number using the `--n_generations`.


Here we generate more examples for the `run_app` skill that opens an App on your Mac.

```bash
python training_data.py -c --file ../examples/run_app_examples.txt
```

The provided files have already been generated using this script.

## Collect training data

Once you have training data generated, we convert it to a json file that will be used for training. You can use the `--files` argument to provide a list of files to convert. The script will print the json to the console, you can redirect it to a file using `> dataset/training_data.json`.

```bash
python training_data.py -t --files ../examples/*examples.txt > dataset/training_data.json
```

## Get base model

Depending on what base model you are using, you may need to request approval from the creator team. 
If you don't have access to Llama2 for instance, you can either use [OpenLlama](https://huggingface.co/openlm-research), [NousResearch LLama2](https://huggingface.co/NousResearch/Llama-2-7b-hf) or [MistralAI](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1). You will have to adapt the formatting of the training data to the model you are using.

## Run fine tuning

You can perform fine-tune using the provided [finetune.ipynb](finetune.ipynb) script. Running this script with everything as default will reproduce our Mistral Instruct experiments.
It uses:
- Peft with qLoRA for efficient finetuning
- SFTTrainer from [trl](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) for data pre-processing
- W&B to log metrics and predictions during training.
- W&B to log the model checkpoint and the corresponding prompt formatting used. We only log the adapters to save memory.

## Eval

You can run evaluation with the `eval.py` script, this will:
- Pull the test dataset from W&B
- Apply the corresponding prompt
- Log the metrics and predictions to W&B

You can override the parameters
- If you pass `MODEL_ID` it will use that model from the HF hub instead of the fine-tuned artifact
- If you pass `PROMPT` it will override the defaults prompts with the one you pass. (long prompts are better passed as a file)
- `PROMPT_FILE`: A better way of passing your prompt is probably using a file, check [prompts](prompts) folder for examples.

> you have to pair the right format with the right model. Llama and Mistral have different formats.