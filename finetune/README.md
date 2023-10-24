# Fine Tuning Steps

## Install requirements

```
cd finetune
pip install -r requirements_finetune.txt
```

## Optionally generate more training data

```
python training_data.py -c --file ../examples/run_app_examples.txt
```

## Collect training data

```
python training_data.py -t --files ../examples/*examples.txt > dataset/training_data.json
```

## Get base model

## Run fine tuning

```
python finetune.py --all
```
