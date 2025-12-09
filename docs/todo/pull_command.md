# fenn pull Command

The `fenn pull` command allows you to download and use pre-made templates from the [fenn templates repository](https://github.com/pyfenn/templates) for your analysis projects.

## Usage

```bash
fenn pull <template> [path] [--force]
```

### Arguments

- `<template>` (required): The name of the template to download. This corresponds to a folder name in the [pyfenn/templates](https://github.com/pyfenn/templates) repository.
- `[path]` (optional): The target directory where the template should be extracted. Defaults to the current directory (`.`).
- `[--force]` (optional): Overwrite existing files in the target directory if it is not empty.

## Examples

### Basic Usage

Download the `base` template into the current directory:

```bash
fenn pull base
```

### Specify Target Directory

Download the `base` template into a specific directory:

```bash
fenn pull base ./my-project
```

### Overwrite Existing Files

If the target directory already contains files, use the `--force` flag to overwrite them:

```bash
fenn pull base ./existing-project --force
```

## Typical Workflow

1. **Choose a template**: Browse available templates at [https://github.com/pyfenn/templates](https://github.com/pyfenn/templates) to find one that matches your needs.

2. **Pull the template**: Run `fenn pull <template-name>` to download the template into your project directory.

3. **Customize**: Modify the downloaded files to fit your specific analysis requirements.

## Available Templates

Check the [templates repository](https://github.com/pyfenn/templates) for the latest list of available templates. Currently, a `base` template is available.

