SMLE: Simplify Machine Learning Environments
============================================

.. image:: https://img.shields.io/github/stars/blkdmr/smle?style=social
   :alt: GitHub stars
.. image:: https://img.shields.io/github/forks/blkdmr/smle?style=social
   :alt: GitHub forks

.. image:: https://img.shields.io/pypi/v/smle
   :alt: PyPI version
.. image:: https://img.shields.io/github/license/blkdmr/smle
   :alt: License
.. image:: https://img.shields.io/pypi/dm/smle.svg?label=downloads&logo=pypi&color=blue
   :target: https://pypi.org/project/smle/
   :alt: PyPI Downloads

**Stop writing boilerplate. Start training.**

SMLE is a lightweight Python framework that automates the "boring stuff" in Machine Learning projects. It handles configuration parsing, logging setup, and experiment tracking so you can focus on the model.

Why SMLE?
=========

* **Auto-Configuration:** ``yaml`` files are automatically parsed and injected into your entrypoint. No more hardcoded hyperparameters.
* **Instant Logging:** All print statements and configs are automatically captured to local logs and remote trackers.
* **Remote Monitoring:** Native integration with `Weights & Biases (WandB) <https://wandb.ai/>`_ to monitor experiments from anywhere.

⚠️ Security & WandB Configuration
---------------------------------

When using the **wandb** section for remote logging, your API key is currently read directly from the ``smle.yaml`` file.

**Crucial:** To prevent exposing your credentials, **do not commit** ``smle.yaml`` to GitHub or remote storage if it contains your real API key.

* **Recommendation:** Add ``smle.yaml`` and ``*.log`` files to your ``.gitignore`` file immediately.
* **Disable:** You can safely remove the ``wandb`` section from the YAML file if you do not need remote logging features.

Installation
============

.. code-block:: bash

    pip install smle

Quick Start
===========

1. Initialize a Project
-----------------------

Run the CLI tool to generate a template and config file:

.. code-block:: bash

    smle init

2. Write Your Code
------------------

Use the ``@app.entrypoint`` decorator. Your configuration variables are automatically passed via ``args``.

.. code-block:: python

    from smle import SMLE

    app = SMLE()

    @app.entrypoint
    def main(args):
        # 'args' contains your smle.yaml configurations
        print(f"Training with learning rate: {args['training']['lr']}")

        # Your logic here...

    if __name__ == "__main__":
        app.run()

3. Run It
---------

.. code-block:: bash

    python main.py

Configuration (``smle.yaml``)
=============================

SMLE relies on a simple YAML structure. You can generate a blank template using:

.. code-block:: bash

    smle create yaml

Contributing
============

Contributions are welcome! If you have ideas for improvements, feel free to fork the repository and submit a pull request.

#. Fork the Project
#. Create your Feature Branch (``git checkout -b feature/AmazingFeature``)
#. Commit your Changes (``git commit -m 'Add some AmazingFeature'``)
#. Push to the Branch (``git push origin feature/AmazingFeature``)
#. Open a Pull Request

Roadmap
=======

🚀 High Priority
----------------

* **Documentation:** Write comprehensive documentation and examples.
* **Security:** Improve user key management (e.g., WandB key) using ``.env`` file support.
* **Configuration:** Add support for multiple/layered YAML files.

🔮 Planned Features
-------------------

* **ML Templates:** Automated creation of standard project structures.
* **Model Tools:** Utilities for Neural Network creation, training, and testing.
* **Notifications:** Email notification system for completed training runs.
* **Data Tools:** Data exploration and visualization helpers.
* **Analysis:** Result analysis tools (diagrams, confusion matrices, etc.).
* **Integrations:** Support for TensorBoard and similar tracking tools.
* **Testing:** Comprehensive unit and integration tests for the framework.
