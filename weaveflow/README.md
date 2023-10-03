## Setup project

Setup a pyenv

```
pyenv virtualenv 3.10.7 text-extraction
pyenv local text-extraction
pip install -r requirements.txt
```

IMPORTANT: now edit `entity` and `project` in settings.py

The next block creates the following:

- The first dataset version
- Two models
- Evaluations of both models
- Three boards: dataset browser, model browser, model comparsion

This prints a bunch of errors, but they are actually benign. It should take about a minute.

```
mkdir -p dataset
cp example_data/* dataset/
python bootstrap_project.py
```

## visit the UI

<!-- Now you can Browse to https://weave.wandb.ai/browse/wandb/<entity&gt;/<project&gt; to find
your boards. -->

Usually, you could view the weave UI at weave.wandb.ai. However, this repo currently relies on unreleased weave and wandb changes, so you need to run the weave server locally.

To do this,

```
git checkout https://wandb.ai/wandb/weave
git checkout eval

# Follow the instructions in DEVELOPMENT.md to get the Weave UI running locally
```

The above prints a link to the UI

You can also see the underlying W&B artifacts by going to https://wandb.ai, browsing to your project, and clicking the artifacts tab.
