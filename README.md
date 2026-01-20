# UAAG2: Uncanonical Amino Acid Generative Model v2

### Overall goal of the project
The project implements an **MLOps pipeline** for **UAAG2** (Uncanonical Amino Acid Generative Model v2), a diffusion-based generative model for protein engineering. While standard models are restricted to 20 natural amino acids, UAAG2 models side chains in full atomic detail as graphs. Our goal is to transition this $E(3)$-equivariant framework into a production system capable of predicting variant effects for non-canonical amino acids (NCAAs).

### Data
* **Protein Data Bank (PDB)**: 1,000 structures are used to extract residues and local environments within a 10 Å radius.
* **NCAA Datasets**: Data from SwissSidechain and Ilardo et al. (2019) provide novel chemical environments.
* **PDBBind**: Protein-ligand complexes are integrated to capture diverse non-covalent interactions.
* **DMS Benchmarks**: Performance is validated against experimental Deep Mutational Scanning data, specifically PUMA and CP2 benchmarks.

### Hugging Face data checks
We mirror the primary dataset in the public Hugging Face repository `yhsure/uaag2-data`. To refresh local data and
generate a basic markdown report, run:

```bash
$ uv run invoke fetch-data
$ uv run python src/uaag2/hf_data_report.py
```

### Model
The architecture is an **$E(3)$-equivariant Graph Neural Network** based on the EQGAT-diff framework, utilizing 7 message-passing layers and 3.6M parameters. It employs a multi-modal diffusion process that perturbs continuous atomic coordinates and discrete categorical features like atom types and formal charges. To accommodate variable atom counts, the model implements a **virtual node strategy**, allowing it to sample side chains of varying sizes within a unified generative paradigm. Denoising ensures Euclidean symmetry, and mutational effects are calculated by comparing sampled likelihoods of mutant versus wild-type residues.

### Nix Development Environment
A `flake.nix` file is provided to manage system-level dependencies (such as Docker, Colima, and the Google Cloud SDK) that are not managed by `uv`. Run `nix develop` to enter a shell with these tools pre-installed.

## Project structure

The directory structure of the project looks like this:
```txt
├── configs/                  # Configuration files
├── data/                     # Data directory

├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile

│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── README.md
│   └── source/
│       └── index.md
├── models/                   # Trained models
│   └── model.pth
├── notebooks/                # Jupyter notebooks
├── old/                      # Legacy code and scripts
├── reports/                  # Reports
│   └── figures/
│       └── training_statistics.png
├── src/                      # Source code
│   └── uaag2/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py

│       ├── model.py
│       ├── train.py
│       └── visualize.py
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── tasks.py                  # Project tasks
└── uv.lock                   # Dependency lock file
```

## Docker images



Project template created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
