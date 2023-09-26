import weave
from weave.panels import panel_board
from weave import ops_domain
from weave import weave_internal

from weaveflow import settings
from weaveflow import cli


def make_board(initial_entity_name: str, initial_project_name: str):
    varbar = panel_board.varbar()

    entity_name_val = varbar.add(
        "entity_name_val", initial_entity_name, hidden=True)
    entity = ops_domain.entity(entity_name_val)
    varbar.add(
        "entity_name",
        weave.panels.Dropdown(
            entity_name_val, choices=ops_domain.viewer().entities().name()
        ),
    )

    project_name_val = varbar.add(
        "project_name_val", initial_project_name, hidden=True)
    project = ops_domain.project(entity_name_val, project_name_val)
    varbar.add(
        "project_name",
        weave.panels.Dropdown(
            project_name_val, choices=entity.projects().name()),
    )

    dataset_name_val = varbar.add("dataset_name_val", "dataset", hidden=True)

    varbar.add(
        "dataset_name",
        weave.panels.Dropdown(
            dataset_name_val, choices=project.artifactType(
                "Dataset").artifacts().name()
        ),
    )

    dataset_version_val = varbar.add("dataset_version_val", "latest")

    dataset_artifact_version = varbar.add(
        "dataset_artifact_version",
        project.artifactVersion(dataset_name_val, dataset_version_val),
    )

    dataset = varbar.add(
        "dataset",
        weave.ops.get(
            weave_internal.const("wandb-artifact:///")
            + entity_name_val
            + "/"
            + project_name_val
            + "/"
            + dataset_name_val
            + ":"
            + dataset_version_val
            + "/obj",
        ),
    )
    print(dir(dataset))
    main = weave.panels.Group(
        layoutMode="grid",
        showExpressions=True,
        enableAddPanel=True,
    )

    main.add(
        "version_count",
        dataset_artifact_version.artifactSequence().versions().count(),
        layout=weave.panels.GroupPanelLayout(x=0, y=0, w=6, h=4),
    )

    main.add(
        "example_count",
        dataset.rows.count(),
        layout=weave.panels.GroupPanelLayout(x=6, y=0, w=6, h=4),
    )
    main.add(
        "wandb_link",
        weave_internal.const("https://wandb.ai/")
        + entity_name_val
        + "/"
        + project_name_val
        + "/artifacts/"
        + dataset_name_val
        + "/"
        + dataset_version_val,
        layout=weave.panels.GroupPanelLayout(x=0, y=4, w=12, h=4),
    )

    main.add(
        "used_by_runs",
        weave.panels.Table(
            dataset_artifact_version.usedBy(),
            columns=[
                lambda run: run.id(),
                lambda run: run.state(),
                lambda run: run.loggedArtifactVersions().count(),
            ],
        ),
        layout=weave.panels.GroupPanelLayout(x=12, y=0, w=12, h=8),
    )

    main.add(
        "table",
        weave.ops.obj_getattr(dataset, "rows"),
        layout=weave.panels.GroupPanelLayout(x=0, y=8, w=24, h=16),
    )

    return weave.panels.Board(vars=varbar, panels=main)


if __name__ == "__main__":
    entity = 'l2k2'
    project = 'otto3'
    # print(settings.entity, settings.project)
    cli.publish(make_board(entity, project), f"datasets")
