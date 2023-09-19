# Fine Tuning Steps

## Install requirements

```
pip install -r requirements_finetune.txt
```

## Collect training data

```
python training_data.py -t --files skills/*examples.txt > training_data.json
```

## Get base model

## Run fine tuning

```
python finetune.py --all
```
