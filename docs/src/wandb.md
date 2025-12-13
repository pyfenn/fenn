# Experiment tracking with Weights & Biases (W&B) in FENN

A step by step guide on how to integrate [Weights & Biases](https://wandb.ai) (W&B) into a FENN‑based project in order to track machine learning experiments (metrics, hyperparameters, models, and training logs).

> **Note**  
> This guide assumes that you have an active W&B account at <https://wandb.ai>.

## 1. Obtain your WandB API key

Before you can log experiments from FENN to W&B, you must create a W&B account and obtain a personal API key associated with that account. This key is used to authenticate all requests from your code to the W&B backend.

1. Log in to <https://wandb.ai> and open your user *Settings* page from the account menu.  
2. Locate the **API Keys** section and create or copy your personal API key, which will be required to authenticate the W&B client from your code.

### ⚠️ Security & WandB Configuration

> **Warning**  
> When using the `wandb` section for remote logging, the API key is read from `fenn.yaml`. To avoid exposing credentials, do not commit `fenn.yaml` or log files with real keys to any public repository.  
> It is a good practice to:
> - Add `fenn.yaml` and `*.log` to `.gitignore`.  
> - Remove the `wandb` section entirely if remote logging is not required.

## 2. Configuring FENN: the `fenn.yaml` file

FENN uses a `fenn.yaml` file to centralize project configuration (training, logging, etc.). To enable W&B, add a dedicated section.

Minimal example of `fenn.yaml` with W&B support:

```
wandb:
entity: your_wandb_account
key: your_wandb_key
```

## 3. Running an experiment

Once the `fenn.yaml` file has been configured, including the `wandb` section, you can start an experiment by running your FENN entrypoint script as a standard Python program:

`python main.py`



During execution, the script reads the configuration from `fenn.yaml`, initializes the W&B client, and sends configuration data and training metrics to your W&B project. When the run completes, you can open <https://wandb.ai>, navigate to the configured project, and inspect dashboards with loss and accuracy curves, stored experiment configurations, and any saved files such as models and logs.
