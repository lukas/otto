import typing
import urllib
import wandb
import weave

from weave import uris
from weave import artifact_local
from weave import artifact_wandb

from weaveflow import settings


def make_wandb_table(list_of_dicts):
    table = wandb.Table(columns=list(list_of_dicts[0].keys()), allow_mixed_types=True)
    for d in list_of_dicts:
        table.add_data(*d.values())
    return table


def wandb_uri_to_weave_uri(wandb_uri: str) -> str:
    if not wandb_uri.startswith("wandb-artifact://"):
        raise ValueError(f"Invalid wandb artifact uri: {wandb_uri}")
    without_scheme = wandb_uri[len("wandb-artifact://") :]
    url_info = urllib.parse.urlparse(without_scheme)
    parts = url_info.path.split("/")
    return f"wandb-artifact:///{parts[0]}/{parts[1]}/{parts[2]}/obj"


def wandb_artifact_to_weave_uri(wandb_artifact: wandb.Artifact) -> uris.WeaveURI:
    name, version = wandb_artifact.name.split(":")
    return artifact_wandb.WeaveWBArtifactURI(
        entity_name=wandb_artifact.entity,
        project_name=wandb_artifact.project,
        name=name,
        version=version,
        path="obj",
    )


def weave_uri_to_wandb_uri_string(weave_uri: uris.WeaveURI) -> str:
    if isinstance(weave_uri, artifact_local.WeaveLocalArtifactURI):
        # Convert it to a string, but don't need to modify it.
        return str(weave_uri)
    if not isinstance(weave_uri, artifact_wandb.WeaveWBArtifactURI):
        raise ValueError(f"Unhandled weave uri: {weave_uri}")
    return f"wandb-artifact://{weave_uri.entity_name}/{weave_uri.project_name}/{weave_uri.name}:{weave_uri.version}"
