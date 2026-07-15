# Contribution Guide

Welcome to the **fenn** project! This guide explains how to prepare your changes, work with Git, and open good pull requests.

## Initial Checks

Before you start coding, read the main `README.md` and the [documentation](https://pyfenn.org/) to understand the project goals, basic usage, and current roadmap. If you are unsure what to work on or want feedback on an idea, start a conversation on the project’s Discord server or in the GitHub discussion thread linked from the README so the maintainers can help you scope a useful contribution.

Whenever possible, prefer small, focused changes over very large pull requests.
If you plan a bigger feature or refactor, discuss it first to confirm that it fits the project direction.

## Basic Git Workflow

If you are new to Git and GitHub, the steps below describe a simple way to contribute:

- Fork the repository to your own GitHub account.
- Clone your fork locally:
  `git clone https://github.com/<your-username>/fenn.git`
- Create a new branch for your work:
  `git checkout -b feature`
- Make and test your changes. We use nox for testing:
  `nox`
- Stage your changes:
  `git add <files>`
- Commit with a clear message:
  `git commit -m "Describe your change"`
- Push your branch:
  `git push origin feature`
- Open a pull request from your branch into the main `fenn` repository.

Try to keep each branch focused on a single issue or feature so that reviews are easier.

## Making Changes and Opening a PR

When you make changes, aim to:

- Follow the existing code style and structure where possible.
- Add or update tests and documentation if your change affects behavior or public APIs.
- Keep commits logically grouped (for example, separate “refactor” from “new feature” where it makes sense).

**Please note** that, in order to test your changes, you need to reinstall `fenn` locally in editable mode along with its dev and test dependencies. `fenn` uses pre-commit, ruff, and typos for automated linting, formatting, and spell checking. From the base project directory (the one containing the project `pyproject.toml` file), run:

```
pip install -e ".[dev,test]"
```

### Set up pre-commit hooks

Run the following command once after cloning your fork to install `pre-commit` and wire up the Git hook:

```bash
python -m pip install pre-commit && pre-commit install
```

This only needs to be done once after cloning your fork.

This makes pre-commit run the configured hooks on every `git commit`. Your commit may be blocked if a hook reports a failure or modifies a file. `ruff format` may reformat your code automatically, so you'll need to review the modifications and stage them again before committing. Without this step your commits may fail CI.

For more information on `ruff` and `typos` see the docs:

- [`ruff` documentation](https://docs.astral.sh/ruff/)
- [`typos` documentation](https://github.com/crate-ci/typos)

Once your branch is ready:

- Ensure the project runs as expected (and tests pass, if available).
- **Rebase or merge the latest default branch into your branch to resolve conflicts before opening the PR.**
- Push your final changes and prepare to open a pull request.

## Submitting a Pull Request (PR)

1. From your fork on GitHub, use the “Compare & pull request” button to create a PR.
2. Use a clear, descriptive title and fill out the PR description so reviewers understand what you changed and why.
3. If you are working on a labeled issue (for example, “good first issue”), reference the issue number in the PR description (for example, “Fixes #123”).
4. Submit the PR and respond to review comments; you may be asked to adjust code, tests, or documentation before merge.

## Reporting Issues and Getting Help

If you find a bug or have a feature request but do not plan to implement it yourself:

- Open a GitHub issue with a clear description of the problem or idea.
- For bugs, describe the expected behavior, the actual behavior, and how to reproduce it (include versions and environment details when possible).
- For feature requests, explain the use case and how it would benefit fenn users.

If you need guidance at any point, use the Discord server or the GitHub discussion thread to ask questions and coordinate with maintainers and other contributors.

## Building the Documentation

Fenn has two documentation sites:

- **[pyfenn.org](https://pyfenn.org)** — User-facing docs built with [MkDocs](https://www.mkdocs.org/), lives in [pyfenn/docs](https://github.com/pyfenn/docs)
- **[api.pyfenn.org](https://api.pyfenn.org)** — Developer API reference built with [Sphinx](https://www.sphinx-doc.org/), lives in [pyfenn/dev-docs](https://github.com/pyfenn/dev-docs)

### User Docs (MkDocs)

Install dependencies:

```bash
pip install mkdocs-material
```

Clone and serve locally:

```bash
git clone https://github.com/pyfenn/docs.git
cd docs
mkdocs serve
```

Open `http://127.0.0.1:8000` in your browser to preview.

If your change affects user-facing features, update the relevant `.md` files in `docs/src/` and open a PR to [pyfenn/docs](https://github.com/pyfenn/docs).

### Developer API Reference (Sphinx)

Install dependencies:

```bash
pip install sphinx furo sphinx-copybutton sphinx-design myst-parser
```

Clone and build locally:

```bash
git clone https://github.com/pyfenn/dev-docs.git
cd dev-docs
make html
```

Open `_build/html/index.html` in your browser to preview.

The API reference at [api.pyfenn.org](https://api.pyfenn.org) is automatically updated when changes are merged into the main fenn repository. If your change adds or modifies public APIs, make sure your docstrings are up to date — the API reference is generated directly from them.

### Contributing to the Docs

If your change affects public APIs or adds new features, please update the relevant `.rst` files in the docs repository and open a separate PR there.
