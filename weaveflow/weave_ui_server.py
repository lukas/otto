import time

from weave import context
from weave import util


def force_is_notebook():
    return True


util.is_notebook = force_is_notebook

# server starts here
context.get_client()


def remove_suffix(input_str, suffix):
    if input_str.endswith(suffix):
        return input_str[: -len(suffix)]
    return input_str


frontend_url = context.get_frontend_url()

print(remove_suffix(frontend_url, "/__frontend/weave_jupyter"))

while True:
    time.sleep(1)
