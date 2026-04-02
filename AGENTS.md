> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, use `uv run invoke --list` or refer to the
    `tasks.py` file.
    * Quick training via invoke: `uv run invoke train --gpus=1 --batch-size=32`
    * With experiment config: `uv run invoke train --experiment=production --gpus=8`
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.
* The project uses `Hydra` for configuration management:
    * Training configurations are in `configs/` directory with YAML files.
    * Direct training command: `uv run python src/uaag2/train.py`
    * To override config: `uv run python src/uaag2/train.py optimizer.lr=0.001 data.batch_size=64`
    * See `docs/HYDRA_CONFIG.md` for detailed usage guide.
    * Both `invoke train` and direct `python src/uaag2/train.py` work - invoke provides convenience wrappers

# GPU Support

The project supports multiple GPU backends:

* **NVIDIA GPUs (CUDA)**: Default installation via `uv sync` installs PyTorch with CUDA 12.8
* **AMD GPUs (ROCm)**: For AMD GPUs (e.g., LUMI supercomputer), use:
  * Quick install: `bash install_rocm.sh`
  * See `docs/ROCM_INSTALLATION.md` for detailed instructions
* **Apple Silicon (MPS)**: Automatically used on macOS with Apple Silicon

Platform-specific installations:
* Linux x86_64 with NVIDIA GPUs: `uv sync` (default, uses CUDA 12.8)
* Linux x86_64 with AMD GPUs: `bash install_rocm.sh` (uses ROCm 6.2)
* macOS: `uv sync` (uses CPU/MPS)
* Linux ARM64: `uv sync` (CPU only)

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Configuration Management

* The project uses Hydra for configuration instead of argparse.
* All training configurations are in YAML format under `configs/` directory.
* Configuration is hierarchical: train.yaml → model/data/diffusion/optimizer configs.
* Use command-line overrides for hyperparameter tuning without modifying files.
* Each training run automatically saves the resolved configuration.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project will be using `mkdocs` for documentation. To build the docs locally, use
    `uv run mkdocs serve`
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
