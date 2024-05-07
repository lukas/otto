## Run fine tuning

You can perform fine-tune using the provided [finetune.py](finetune.py) script. Running this script with everything as default will reproduce our Mistral Instruct experiments.

Or you can run
```bash
python finetune.py
```

It uses:
- Peft with qLoRA for efficient finetuning
- SFTTrainer from [trl](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) for data pre-processing
- W&B to log metrics and predictions during training.
- W&B to log the model checkpoint and the corresponding prompt formatting used. We only log the adapters to save memory.

## Eval

You can run evaluation with the `eval.py` script, this will:
- Pull the test dataset from W&B
- Apply the corresponding prompt
- Log the metrics and predictions to W&B Weave