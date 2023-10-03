import glob
import json
import os

import weave

from weaveflow import settings
from weaveflow import base_types
from weaveflow import cli

from finetune import training_data


def read_dataset():
    print("Reading Dataset")
    dataset_rows = []
    for i, row in enumerate(open("dataset/training_data.json")):
        dataset_rows.append({"id": i, **json.loads(row)})
    print("DR ", dataset_rows[:100])
    print(base_types.Dataset(rows=weave.WeaveList(dataset_rows[:100])))
    return base_types.Dataset(rows=weave.WeaveList(dataset_rows[:100]))


def publish():
    dataset = read_dataset()
    return cli.publish(dataset, "dataset")


if __name__ == "__main__":
    publish()
