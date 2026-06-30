# Fenn: Friendly Environment for Neural Networks

<p align="center"><img src="banner.png" alt="fenn logo" width="1000"></p>

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/pyfenn/fenn?style=social) ![GitHub forks](https://img.shields.io/github/forks/pyfenn/fenn?style=social) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/261c40f69583462baa200aee959bcc8f)](https://app.codacy.com/gh/blkdmr/fenn/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![codecov](https://codecov.io/gh/pyfenn/fenn/graph/badge.svg?token=7RTTZ1SFMM)](https://codecov.io/gh/pyfenn/fenn) ![PyPI version](https://img.shields.io/pypi/v/fenn) ![License](https://img.shields.io/github/license/pyfenn/fenn) [![PyPI Downloads](https://img.shields.io/pypi/dm/fenn.svg?label=downloads&logo=pypi&color=blue)](https://pypi.org/project/fenn/) [![Discord Server](https://img.shields.io/badge/Discord-PyFenn-5865F2?logo=discord&logoColor=white)](https://discord.com/invite/6v9xtJxvN7) [![Sponsor](https://img.shields.io/badge/sponsor-GitHub-pink)](https://github.com/sponsors/blkdmr)

</div>

**The open engine for deep learning workflows.**

Friendly Environment for Neural Networks (fenn) is a simple framework that automates ML/DL workflows by providing prebuilt trainers, templates, logging, configuration management, and much more. With fenn, you can focus on your model and data while it takes care of the rest.

## Support fenn

If fenn is useful for your work or research, consider supporting its development.

You can support the project by **starring the repository** on GitHub. It improves visibility and helps others discover fenn.

Sponsorship also helps fund maintenance, improvements, and new features.

Support the project:
https://github.com/sponsors/blkdmr

## Why fenn?

- **Auto-Configuration**: YAML files are automatically parsed and injected into your entrypoint with CLI override support. No more hardcoded hyperparameters or scattered config logic.

- **Unified Logging**: All logs, print statements, and experiment metadata are automatically captured to local files and remote tracking backends simultaneously with no manual setup required.

- **Backend Monitoring**: Native integration with industry-standard trackers like [Weights & Biases](https://wandb.ai/) (W&B) for centralized experiment tracking and [TensorBoard](https://www.tensorflow.org/tensorboard) for real-time metric visualization

- **Instant Notifications**: Get real-time alerts on **Discord** and **Telegram** when experiments start, complete, or fail—no polling or manual checks.

- **Trainers**: Built-in support for training loops, validation, and testing with minimal boilerplate. Just define your model and data, and let fenn handle the rest.

- **Template Ready**: Built-in support for reproducible, shareable experiment templates.


## Quickstart

Install the fenn library using

```bash
pip install fenn
```

or

```bash
uv pip install fenn
```

### Initialize a Project

Use the CLI to discover and download a project template.

#### 1. List available templates

```bash
fenn list
```

This fetches the directory listing from [`pyfenn/templates`](https://github.com/pyfenn/templates) and prints the templates you can use.

#### 2. Pull a template

```bash
fenn pull <template> [path]
```

Examples:

```bash
fenn pull empty            # pull into the current directory
fenn pull empty ./my-proj  # pull into ./my-proj (created if missing)
```

Each template ships at least a `main.py` entrypoint and a `fenn.yaml` configuration file in the target directory. Most templates also include a `README.md`, a `requirements.txt`, and a `modules/` directory with example model and dataset code.

To avoid clobbering work, `fenn pull` refuses to write into a non-empty target directory. Pass `--force` to overwrite existing files:

```bash
fenn pull empty --force
```

Hidden entries (those starting with `.`, such as `.git`) do not count as "non-empty".

#### 3. Customize and run

Open the generated `fenn.yaml` and adjust hyperparameters, paths, logging, and integrations for your project (see [Configuration](#configuration) below). Then run the entrypoint:

```bash
python main.py
```

#### Common issues

- **`Template <name> not found`** — The template name doesn't match a directory in [`pyfenn/templates`](https://github.com/pyfenn/templates). Run `fenn list` to see valid names.
- **`Refusing to pull into non-empty directory`** — Either pull into an empty directory, point `path` at a fresh one, or pass `--force` to overwrite.
- **`Network error` / `Failed to check template existence`** — Check connectivity. The CLI uses the unauthenticated GitHub API to look up and download templates, which is subject to GitHub's rate limit.
- **`fenn: command not found` after installation** — Your Python scripts directory may not be on your `PATH`. Try running with `python -m fenn` instead, or add the scripts directory to your PATH. On most systems: `export PATH="$HOME/.local/bin:$PATH"`.
- **`fenn.yaml not found` when running `main.py`** — Make sure you are running the script from the same directory that contains `fenn.yaml`. fenn looks for the config file in the current working directory by default.
- **`KeyError` on `args['section']['key']`** — The key referenced in your code does not exist in `fenn.yaml`. Double-check spelling in both files. YAML is case-sensitive.
- **`ModuleNotFoundError` after pulling a template** — `fenn` automatically attempts to install template dependencies during the pull process. If an environment issue prevents this, navigate into your project directory and run `pip install -r requirements.txt` manually.
- **GitHub API rate limit exceeded during `fenn list` or `fenn pull`** — The unauthenticated GitHub API allows 60 requests/hour per IP. Wait a few minutes and try again, or set a `GITHUB_TOKEN` environment variable if your fenn version supports authenticated requests.

### Configuration

fenn relies on a simple YAML structure to define hyperparameters, paths, logging options, and integrations. You can configure the ``fenn.yaml`` file with the hyperparameters and options for your project.

The structure of the ``fenn.yaml`` file is:

```yaml
# ---------------------------------------
# Fenn Configuration (Modify Carefully)
# ---------------------------------------

project: empty

# ---------------------------
# Logging & Tracking
# ---------------------------

logger:
  dir: logger

export:
  dir: exports

# ---------------------------------------
# Example of User Section
# ---------------------------------------

train:
    lr: 0.001
```

### Write Your Code

Use the `@app.entrypoint` decorator. Your configuration variables are automatically passed via `args`.

```python
from fenn import Fenn

app = Fenn()

@app.entrypoint
def main(args):
    # 'args' contains your fenn.yaml configurations
    print(f"Training with learning rate: {args['train']['lr']}")

    # Your logic here...

if __name__ == "__main__":
    app.run()
```

By default, fenn will look for a configuration file named `fenn.yaml` in the current directory. If you would like to use a different name, a different location, or have multiple configuration files for different configurations, you can call `set_config_file()` and update the path or the name of your configuration file. You must assign the filename before calling `run()`.

The optional `export.dir` setting centralizes where artifacts are written. Components that export files can use this shared directory instead of requiring an output path to be passed through every call.

```python
app = Fenn()
app.set_config_file("my_file.yaml")
```

### Run It

You can run your code as usual

```bash
python main.py
```

and fenn will take care of the rest for you.

### Training Models

Use built-in trainers to handle your training loops with minimal boilerplate.

```python
import torch.nn as nn
import torch.optim as optim

from fenn.nn.trainers import ClassificationTrainer
from fenn.nn.utils import Checkpoint

@app.entrypoint
def main(args):

    # Define your data
    train_loader = DataLoader(train_dataset, batch_size=args["train"]["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["test"]["batch"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args["test"]["batch"], shuffle=False)

    # Define your model
    model = nn.Sequential( ... )
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=float(args["train"]["lr"]))

    # Initialize a ClassificationTrainer
    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss,
        optim=optimizer,
        num_classes=4
    )

    # Train and predict your model
    trainer.fit(train_loader, epochs=10, val_loader=val_loader)
    preds = trainer.predict(test_loader)
```

## CLI Reference

A quick reference for all available fenn CLI commands.

| Command | Description |
|---|---|
| `fenn dashboard` | Launch the local log-browser web UI |
| `fenn grid <path>` | By setting grid/train section in template, you can run a Fenn project several times, with all possible grid hyperparams. Also, it is possible to specify path to main.py file (e.g. my_temp/main.py) |
| `fenn list` | List all available templates from [`pyfenn/templates`](https://github.com/pyfenn/templates) |
| `fenn pull <template>` | Pull a template into the current directory |
| `fenn pull <template> <path>` | Pull a template into the specified path (created if missing) |
| `fenn pull <template> --force` | Pull a template and overwrite existing files |

## Contributing

Contributions are welcome!

Interested in contributing? Join the community on [Discord](https://discord.com/invite/6v9xtJxvN7).

We can then discuss a possible contribution together, answer any questions, and help you get started!

**Please consult our CONTRIBUTING.md and CODE_OF_CONDUCT.md before opening a pull request.**

## Maintainers

The development and long-term direction of **fenn** is guided by the following maintainers:

| Maintainer | Role |
|------------|------|
| [@blkdmr](https://github.com/blkdmr) | Creator & Project Administrator |
| [@giuliaOddi](https://github.com/giuliaOddi) | Project Administrator |
| [@ApusBerliozi](https://github.com/ApusBerliozi) | Project Administrator |

Maintainers oversee the project roadmap, review pull requests, coordinate releases, and ensure the long-term stability and quality of the framework.

Thank you for supporting the project.
