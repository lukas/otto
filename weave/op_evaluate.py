import dataclasses
import weave
import typing
import time

import pandas as pd

from weave import ops_arrow

from weaveflow import base_types
from weaveflow import cli


def summarize_examples(label: weave.WeaveList, output: weave.WeaveList) -> pd.DataFrame:
    label = label.to_pandas()
    output = output.to_pandas()
    # Initialize summary DataFrame
    result = pd.DataFrame(index=output.index)

    # For now, make this 0 or 1 instead of True or False,
    # just because PanelFacet in board_model_compare doesnt handle the boolean
    # TODO: fix.
    result["correct"] = (output == label).astype(int)

    return ops_arrow.dataframe_to_arrow(result)


def summarize(eval_table: weave.WeaveList) -> dict:
    return {
        "accuracy": eval_table.column("correct").to_pandas().sum() / len(eval_table),
    }


@weave.op(input_type={"model": weave.types.ObjectType()})
def evaluate_correct(dataset: base_types.Dataset, model) -> typing.Any:
    result = []
    latencies = []
    for row in dataset.rows:
        start_time = time.time()
        result.append(weave.use(model.predict(row["user"])))
        latencies.append(time.time() - start_time)
    output = weave.WeaveList(result)
    label = dataset.rows.column("answer")
    example_summary = summarize_examples(label, output)
    eval_table = weave.WeaveList(
        {
            "dataset_id": dataset.rows.column("id"),
            "output": output,
            "latency": weave.WeaveList(latencies),
            "summary": example_summary,
        }
    )
    return {
        "eval_table": eval_table,
        "summary": summarize(eval_table.column("summary")),
    }


def main():
    cli.weave_op_main(evaluate_correct)


if __name__ == "__main__":
    main()
