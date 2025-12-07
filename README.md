# fenn: Friendly Environment for Neural Networks

<p align="center"><img src="logo.png" alt="fenn logo" width="1000"></p>

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/blkdmr/fenn?style=social) ![GitHub forks](https://img.shields.io/github/forks/blkdmr/fenn?style=social) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/261c40f69583462baa200aee959bcc8f)](https://app.codacy.com/gh/blkdmr/fenn/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) ![PyPI version](https://img.shields.io/pypi/v/fenn) ![License](https://img.shields.io/github/license/blkdmr/fenn) [![PyPI Downloads](https://img.shields.io/pypi/dm/fenn.svg?label=downloads&logo=pypi&color=blue)](https://pypi.org/project/fenn/) [![Discord Server](https://img.shields.io/badge/Discord-PyFenn-5865F2?logo=discord&logoColor=white)](https://discord.gg/WxDkvktBAa)

</div>

**Stop writing boilerplate. Start training.**

fenn is a lightweight Python framework that automates the "boring stuff" in Machine Learning projects. It handles configuration parsing, logging setup, and experiment tracking so you can focus on the model.

## Why fenn?

* **Auto-Configuration:** `yaml` files are automatically parsed and injected into your entrypoint. No more hardcoded hyperparameters.
* **Instant Logging:** All print statements and configs are automatically captured to local logs and remote trackers.
* **Remote Monitoring:** Native integration with [Weights & Biases (WandB)](https://wandb.ai/) to monitor experiments from anywhere.

## Installation

```bash
pip install fenn
````

## Quickstart

### Initialize a Project

Run the CLI tool to download a template and config file:

```bash
fenn pull base
```

### Configuration

fenn relies on a simple YAML structure to define hyperparameters, paths, logging options, and integrations. You can configure the ``fenn.yaml`` file with the hyperparameters and options for your project.

The structure of the ``fenn.yaml`` file is:

```yaml
# ---------------------------------------
# Fenn Configuration (Modify Carefully)
# ---------------------------------------

project: project_name

# ---------------------------
# Logging & Tracking
# ---------------------------

logger:
  dir: logger

wandb:
  entity: your_wandb_account

# ---------------------------------------
# Example of User Section
# ---------------------------------------

seed: seed
device: 'cpu'/'cuda'

training:
    epochs: n_epochs
    lr: lr
    weight_decay: wd
    batch: batch_size

testing:
    batch: batch_size
```

**Note.** fenn expects your Weights and Biases API key to be in the environment variable `WANDB_API_KEY`. You can put it in the `.env` file, but ensure `.env` is in your `.gitignore`.

### Write Your Code

Use the `@app.entrypoint` decorator. Your configuration variables are automatically passed via `args`.

```python
from fenn import FENN

app = FENN()

@app.entrypoint
def main(args):
    # 'args' contains your fenn.yaml configurations
    print(f"Training with learning rate: {args['training']['lr']}")

    # Your logic here...

if __name__ == "__main__":
    app.run()
```

By default, fenn will look for a configuration file named `fenn.yaml` in the current directory. If you would like to use a different name, a different location, or have multiple configuration files for different configurations, you can set the `config_file` property of fenn to the path of your file. You must assign the filename before calling `run()`.

```python
app = FENN()
app.config_file = "my_file.yaml"
...
app.run()
```

### Run It

```bash
python main.py
```

## Contributing

Contributions are welcome!

If you’re interested in helping, please feel free to join our [discord server](https://discord.gg/WxDkvktBAa) or the dedicated
[discussion page](https://github.com/blkdmr/fenn/discussions/11) and ping there your availability.

We can then discuss a possible contribution together, answer any questions, and help you get started!

**Please, before opening a pull request, consult our CONTRIBUTING.md**

Thank you for your support!

## Roadmap

### High Priority

- **Documentation:** Write comprehensive documentation and examples.

### Planned Features

- **ML Templates:** Automated creation of standard project structures.
- **Model Tools:** Utilities for Neural Network creation, training, and testing.
- **Notifications:** Email notification system for completed training runs.
- **Data Tools:** Data exploration and visualization helpers.
- **Analysis:** Result analysis tools (diagrams, confusion matrices, etc.).
- **Integrations:** Support for TensorBoard and similar tracking tools.
- **Testing:** Comprehensive unit and integration tests for the framework.

