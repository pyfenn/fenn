
#Configuration

You can configure the `fenn.yaml` file with the hyperparameters and options for your project. The structure of the `fenn.yaml` file is:

```
---------------------------------------
FENN Configuration (Modify Carefully)
---------------------------------------
project: project_name

---------------------------
Logging & Tracking
---------------------------
logger:
dir: logger

wandb:
entity: your_wandb_account

---------------------------------------
Example of User Section
---------------------------------------
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

> **Tip**  
> You can add new sections as needed (for example, `model`, `optimizer`, `scheduler`) and access them via the same keys in `args`. This makes it easy to manage multiple experiments by maintaining different YAML files.

## Configuration File Name

By default, FENN will look for a configuration file named `fenn.yaml` in the current directory. If you would like to use a different name, a different location, or have multiple configuration files for different configurations, you can set the `config_file` property of FENN to the path of your file. You must assign the filename before calling `run()`:


```
app = FENN()
app.config_file = "my_file.yaml"
...
app.run()
```