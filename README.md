# Embedding Scope

## Table of Contents

- [Getting Started](#getting-started)

## Getting Started

1. **Install Conda**:

    If Conda isn't installed, download it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

    Conda is required to manage dependencies and ensure a consistent environment.

2. **Create Environment**:

    ```sh
    conda env create -f environment.yml
    ```

3. **Activate Environment**:

    ```sh
    conda activate scope
    ```

4. **Configure Workspace**:

    To configure the workspace where the datasets and trained checkpoints are stored, update the `source/__init__.py` file with the desired path.

    For example:

    ```python
    from pathlib import Path

    workspace = Path("/your/path/to/workspace")
    workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
    ```

You're all set to begin!
