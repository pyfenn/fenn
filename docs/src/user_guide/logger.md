# Logger

The `Logger` class centralizes console logging, file logging, and optional Weights & Biases (WandB) tracking for a FENN session, using the same configuration dictionary loaded from your YAML file. It automatically captures all `print` output, writes a clean, timestamped version to disk, and keeps output in the terminal for interactive use.

## Initialization

In normal usage, the logger is created and managed internally by FENN. Users do not need to import `Logger`, instantiate it, or call its methods directly; it is automatically initialized at the beginning of a FENN run based on the configuration loaded from `fenn.yaml`.

## Log file behavior

When FENN starts a run, the logger:

- Creates or empties a log file named `<project>_<session_id>.log` inside `logger.dir`.
- Prints a short status line to the console indicating the log file name and directory.
- Replaces the built-in [`print()`](https://docs.python.org/3/library/functions.html#print) with an internal wrapper so that all subsequent `print(...)` calls are intercepted.

Example of a log file named `<project>_<session_id>.log`:

```
[2025-11-25 08:48:04] project: project_name
[2025-11-25 08:48:04] training/seed: seed
[2025-11-25 08:48:04] training/epochs: n_epochs
[2025-11-25 08:48:04] training/learning_rate: lr
[2025-11-25 08:48:04] training/weight_decay: wd
[2025-11-25 08:48:04] training/train_batch: batch_size
[2025-11-25 08:48:04] training/test_batch: batch_size
```
