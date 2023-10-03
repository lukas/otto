import weave
from weave.panels import panel_board
from weave import ops_domain
from weave import weave_internal

from weaveflow import settings
from weaveflow import cli


def make_board(
    initial_entity_name: str, initial_project_name: str, predictions_stream_name: str
):
    varbar = panel_board.varbar()

    entity_name_val = varbar.add("entity_name_val", initial_entity_name, hidden=True)
    entity = ops_domain.entity(entity_name_val)
    varbar.add(
        "entity_name",
        weave.panels.Dropdown(
            entity_name_val, choices=ops_domain.viewer().entities().name()
        ),
    )

    project_name_val = varbar.add("project_name_val", initial_project_name, hidden=True)
    project = ops_domain.project(entity_name_val, project_name_val)
    varbar.add(
        "project_name",
        weave.panels.Dropdown(project_name_val, choices=entity.projects().name()),
    )

    model_name_val = varbar.add("model_name_val", "PredictBasic", hidden=True)

    varbar.add(
        "model_name",
        weave.panels.Dropdown(
            model_name_val,
            choices=project.artifactType("PredictBasic").artifacts().name(),
        ),
    )

    model_version_val = varbar.add("model_version_val", "v0")

    model_artifact_version = varbar.add(
        "model_artifact_version",
        project.artifactVersion("PredictBasic", model_version_val),
    )

    model_uri = varbar.add(
        "model_uri",
        weave_internal.const("wandb-artifact:///")
        + entity_name_val
        + "/"
        + project_name_val
        + "/"
        + model_name_val
        + ":"
        + model_version_val
        + "/obj",
        hidden=True,
    )

    model = varbar.add(
        "model",
        weave.ops.get(
            weave_internal.const("wandb-artifact:///")
            + entity_name_val
            + "/"
            + project_name_val
            + "/"
            + model_name_val
            + ":"
            + model_version_val
            + "/obj"
        ),
    )

    prediction_stream = varbar.add(
        "prediction_stream",
        weave.ops.get(
            weave_internal.const("wandb-artifact:///")
            + entity_name_val
            + "/"
            + project_name_val
            + "/"
            + predictions_stream_name
            + ":latest/obj"
        ).rows()
        # There is a frontend refinement issue that causes the filtered
        # panel data to dissapear after refinement. Commenting out
        # for now, we who predictions across all models for the moment.
        # .filter(lambda row: row["inputs.self"] == model_uri),
    )

    main = panel_board.main()

    main.add(
        "model",
        model,
        layout=weave.panels.GroupPanelLayout(x=0, y=0, w=12, h=12),
    )

    main.add(
        "accuracy",
        model_artifact_version.usedBy()
        .filter(lambda run: run.state() == "finished")[-1]
        .summary()["summary.accuracy"],
        layout=weave.panels.GroupPanelLayout(x=12, y=0, w=4, h=4),
    )

    main.add(
        "used_by_runs",
        weave.panels.Table(
            model_artifact_version.usedBy(),
            columns=[
                lambda run: run.id(),
                lambda run: run.state(),
                lambda run: run.loggedArtifactVersions().count(),
            ],
        ),
        layout=weave.panels.GroupPanelLayout(x=12, y=4, w=12, h=8),
    )

    main.add(
        "predictions",
        prediction_stream,
        layout=weave.panels.GroupPanelLayout(x=0, y=12, w=24, h=12),
    )

    return weave.panels.Board(vars=varbar, panels=main)


if __name__ == "__main__":
    cli.publish(
        make_board(settings.entity, settings.project, settings.predictions_stream_name),
        f"datasets",
    )
