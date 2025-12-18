Welcome to **fenn** (Friendly Environment for Neural Networks), a lightweight Python framework designed to strip away the repetitive boilerplate of Machine Learning development.

**Stop writing boilerplate. Start training.**

FENN is a lightweight Python framework that automates the **boring stuff** in Machine Learning projects so you can focus on the model. It handles configuration parsing, logging setup, and experiment tracking in a minimal, opinionated way.

---

## Why fenn?

In a typical ML project, developers often spend hours setting up logging directories, writing YAML parsers, and manually connecting experiment trackers. Fenn automates this entire lifecycle:

* **Auto-Configuration**: YAML files are automatically parsed and injected into your entrypoint. You get full CLI override support without writing a single line of `argparse`.
* **Unified Logging**: All logs, print statements, and experiment metadata are captured to local files and remote backends simultaneously.
* **Multi-Backend Monitoring**: Native integration with [Weights & Biases (W&B)](https://wandb.ai/) and [TensorBoard](https://www.tensorflow.org/tensorboard).
* **Instant Notifications**: Get real-time alerts on **Discord** and **Telegram** when experiments start, finish, or crash.
* **Template Ready**: Download and use reproducible experiment templates to jumpstart new projects.

---

## How it Works

Fenn uses a decorator-based approach to wrap your code in a managed environment.

### Define your configuration

Define your hyperparameters and project settings in a simple YAML structure.

```yaml
project: my_awesome_model

logger:
  dir: logs

train:
  lr: 0.001
  batch_size: 32

```

### Set the entrypoint

Use the `@app.entrypoint` decorator. Fenn injects your YAML settings directly into the `args` variable.

```python
from fenn import FENN

app = FENN()

@app.entrypoint
def main(args):
    # 'args' contains your fenn.yaml configurations automatically
    print(f"Training with learning rate: {args['train']['lr']}")

if __name__ == "__main__":
    app.run()

```

---

## Installation

Get started quickly via pip:

```bash
pip install fenn

```

## Next Steps

To dive deeper into Fenn and start your first project, we recommend following these guides:

| Guide | Description |
| --- | --- |
| **[Quickstart](https://www.google.com/search?q=quickstart.md)** | Go from zero to a running experiment in 5 minutes. |
| **[Configuration](https://www.google.com/search?q=config.md)** | Learn how to structure your YAML and override settings via CLI. |
| **[Integrations](https://www.google.com/search?q=integrations.md)** | Set up W&B, Discord notifications, and TensorBoard. |
| **[CLI Reference](https://www.google.com/search?q=cli.md)** | Master the `fenn pull` and project initialization tools. |