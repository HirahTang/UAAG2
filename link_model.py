"""Script to link a model artifact with aliases in the W&B model registry."""

import os
from typing import Annotated

import typer
import wandb


def link_model(
    artifact_path: str,
    aliases: Annotated[list[str], typer.Option("--alias", "-a")] = ["staging"],
    target_path: Annotated[str | None, typer.Option("--target-path", "-t")] = None,
) -> None:
    """Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with.
        target_path: Optional target path in the registry. If not provided,
            it will be inferred from the artifact_path.

    Example:
        python link_model.py entity/project/artifact_name:version -a staging -a best
    """
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    if not target_path:
        # Extract target path by stripping version if it's already a full path
        # e.g., "entity/project/name:v1" -> "entity/project/name"
        if ":" in artifact_path:
            target_path = artifact_path.split(":")[0]
        else:
            # Fallback for simple names, use environment variables
            target_path = f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_path}"

    artifact = api.artifact(artifact_path)
    artifact.link(target_path=target_path, aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {target_path} with aliases {aliases}")


if __name__ == "__main__":
    typer.run(link_model)
