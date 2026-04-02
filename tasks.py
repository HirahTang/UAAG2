import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "uaag2"
PYTHON_VERSION = "3.12"


# Project commands
@task
def fetch_data(ctx: Context, data_dir: str = "data", force: bool = False) -> None:
    """Fetch protein data from Hugging Face Hub."""
    # Note: We inline the fetch logic here to avoid importing uaag2 package,
    # which requires data files to exist during import.
    fetch_script = f"""
from pathlib import Path
from huggingface_hub import hf_hub_download

HUGGINGFACE_REPO_ID = "yhsure/uaag2-data"
data_path = Path("{data_dir}")
data_path.mkdir(parents=True, exist_ok=True)
force = {force}

files = [
    "aa_graph.json",
    "statistic.pkl",
    "pdb_subset.lmdb",
    "benchmarks/ENVZ_ECOLI.pt",
]

print(f"Fetching data from Hugging Face: {{HUGGINGFACE_REPO_ID}}")
for filename in files:
    local_path = data_path / filename
    if local_path.exists() and not force:
        print(f"  Skipping {{filename}} (already exists)")
        continue
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {{filename}}...")
    hf_hub_download(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=data_path,
    )
    print(f"  Downloaded {{filename}}")
print("Data fetch complete!")
"""
    ctx.run(f"uv run python -c '{fetch_script}'", echo=True, pty=not WINDOWS)


@task
def fetch_model(ctx: Context, artifact: str, output: str = "models/good_model") -> None:
    """Fetch model artifact from W&B."""
    fetch_script = f"""
import os
import shutil
import wandb
from pathlib import Path

artifact_path = "{artifact}"
output_path = "{output}"

api = wandb.Api()
try:
    print(f"Downloading artifact {{artifact_path}}...")
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to {{artifact_dir}}")

    target_dir = Path(output_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    found = False
    # Only the first checkpoint file found is used
    for checkpoint_file in Path(artifact_dir).rglob("*.ckpt"):
        print(f"Found checkpoint: {{checkpoint_file}}")
        target_file = target_dir / checkpoint_file.name
        shutil.move(checkpoint_file, target_file)
        print(f"Moved to: {{target_file}}")
        found = True
        break

    if not found:
        print(f"No .ckpt file found in {{artifact_dir}}")
        exit(1)

except Exception as e:
    print(f"Failed to fetch model: {{e}}")
    exit(1)
"""
    ctx.run(f"uv run python -c '{fetch_script}'", echo=True, pty=not WINDOWS)


@task
def train(
    ctx: Context,
    experiment: str = "",
    batch_size: int = 8,
    num_epochs: int = 50,
    train_size: float = 0.99,
    test_size: int = 32,
    lr: float = 5e-4,
    max_virtual_nodes: int = 5,
    mask_rate: float = 0.0,
    num_layers: int = 7,
    gpus: int = 1,
    logger_type: str = "wandb",
    data_path: str = "data/pdb_subset.lmdb",
    data_info_path: str = "data/statistic.pkl",
    pdbbind_weight: float = 10.0,
    use_metadata_sampler: bool = False,
) -> None:
    """Train model using Hydra configuration.

    Args:
        experiment: Optional experiment config (e.g., 'production', 'quick_test')
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        train_size: Fraction of data to use for training
        test_size: Number of samples for testing
        lr: Learning rate
        max_virtual_nodes: Maximum number of virtual nodes
        mask_rate: Masking rate for data augmentation
        num_layers: Number of model layers
        gpus: Number of GPUs to use
        logger_type: Logger type ('wandb' or 'tensorboard')
        data_path: Path to training data
        data_info_path: Path to data info file
        pdbbind_weight: Weight for PDBBind dataset
        use_metadata_sampler: Whether to use metadata sampler
    """
    # Build Hydra overrides
    overrides = []

    # Add experiment config if specified
    if experiment:
        overrides.append(f"+experiment={experiment}")

    # Add parameter overrides (only non-default values to keep command clean)
    overrides.extend(
        [
            f"data.batch_size={batch_size}",
            f"gpus={gpus}",
            f"num_epochs={num_epochs}",
            f"data.train_size={train_size}",
            f"data.test_size={test_size}",
            f"optimizer.lr={lr}",
            f"data.mask_rate={mask_rate}",
            f"model.max_virtual_nodes={max_virtual_nodes}",
            f"model.num_layers={num_layers}",
            f"data.pdbbind_weight={pdbbind_weight}",
            f"data.training_data={data_path}",
            f"data.data_info_path={data_info_path}",
            f"logger_type={logger_type}",
            f"data.use_metadata_sampler={use_metadata_sampler}",
        ]
    )

    # Build command
    cmd = f"uv run python src/{PROJECT_NAME}/train.py " + " ".join(overrides)

    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def evaluate(
    ctx: Context,
    load_ckpt: str = "",
    benchmark_path: str = "data/benchmarks/ENVZ_ECOLI.pt",
    data_info_path: str = "data/statistic.pkl",
    save_dir: str = "ProteinGymSampling",
    batch_size: int = 32,
    num_samples: int = 500,
    virtual_node_size: int = 15,
    split_index: int = 0,
    num_workers: int = 4,
    num_layers: int = 7,
    logger_type: str = "wandb",
    experiment_id: str = "",
) -> None:
    """Evaluate model on benchmark."""
    import datetime

    if not load_ckpt:
        raise ValueError("--load-ckpt is required for evaluation")

    if not experiment_id:
        experiment_id = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py "
        f"--load-ckpt {load_ckpt} "
        f"--benchmark-path {benchmark_path} "
        f"--data_info_path {data_info_path} "
        f"--save-dir {save_dir} "
        f"--batch-size {batch_size} "
        f"--num-samples {num_samples} "
        f"--virtual_node_size {virtual_node_size} "
        f"--split_index {split_index} "
        f"--num-workers {num_workers} "
        f"--num-layers {num_layers} "
        f"--logger-type {logger_type} "
        f"--id {experiment_id}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain", nix: bool = False, gpu: bool = False) -> None:
    """Build docker images."""

    def with_nix(s):
        return f"nix develop --command bash -c '{s}'" if nix else s

    ctx.run(
        with_nix(
            f"docker build {'--platform linux/amd64' if gpu else ''} -t train{'-gpu' if gpu else ''}:latest . -f dockerfiles/train{'_gpu' if gpu else ''}.dockerfile --progress={progress}"
        ),
        echo=True,
        pty=not WINDOWS,
    )
