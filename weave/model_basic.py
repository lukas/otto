import typing
import functools
import weave

from weaveflow import base_types
from weaveflow import cli


class BasicModelConfig(typing.TypedDict):
    pass


# Model
@weave.type()
class BasicModel(base_types.Model):
    config: BasicModelConfig

    @weave.op()
    def predict(self, user: str) -> str:
        return "other()"


if __name__ == "__main__":
    cli.weave_object_main(BasicModel)
