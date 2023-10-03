import subprocess

import weave
from weave import monitoring

import publish_dataset

import board_datasets
import board_models
import board_model_compare

import model_basic
import model_hf

from weaveflow import settings


def main():
    dataset_ref = weave.storage.publish(
        publish_dataset.read_dataset(), f"{settings.project}/dataset"
    )
    print("published dataset")

    weave.publish(
        board_datasets.make_board(settings.entity, settings.project),
        f"{settings.project}/datasets",
    )
    print("published datasets board")

    model_a_ref = weave.storage.publish(
        model_basic.BasicModel({}),
        f"{settings.project}/PredictBasic",
    )

    # Make some predictions to ensure we have some in the prediction
    # StreamTable
    monitoring.init_monitor(
        f"{settings.entity}/{settings.project}/{settings.predictions_stream_name}"
    )
    model_a = weave.storage.get(model_a_ref)
    weave.use(model_a.predict("hello"))
    weave.use(model_a.predict("my name is jim"))

    print("published model a")
    subprocess.run(
        [
            "python",
            "op_evaluate.py",
            "--dataset",
            str(dataset_ref),
            "--model",
            str(model_a_ref),
        ],
        check=True,
    )
    print("evaluated model a")

    model_b_ref = weave.storage.publish(
        model_hf.HFModel({"model_name": "bert-base-uncased"}),
        f"{settings.project}/PredictBasic",
    )

    # Make predictions
    model_b = weave.storage.get(model_a_ref)
    weave.use(model_b.predict("my name is bob"))

    print("published model b")
    subprocess.run(
        [
            "python",
            "op_evaluate.py",
            "--dataset",
            str(dataset_ref),
            "--model",
            str(model_b_ref),
        ],
        check=True,
    )
    print("evaluated model b")

    # model_c = weave.publish(
    #     model_basic({"shares_skip_chars": 5, "name_up_to_period": True}), "ModelBasic"
    # )
    # cli.run_op(op_evaluate.evaluate_multi_task_f1(dataset, model_c))

    weave.publish(
        board_models.make_board(
            settings.entity, settings.project, settings.predictions_stream_name
        ),
        f"{settings.project}/models",
    )
    print("published models board")

    weave.publish(
        board_model_compare.make_board(settings.entity, settings.project),
        f"{settings.project}/model_compare",
    )
    print("published model compare board")


if __name__ == "__main__":
    main()
