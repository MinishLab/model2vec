import logging
from pathlib import Path

import huggingface_hub
from huggingface_hub.constants import HF_HUB_CACHE

logger = logging.getLogger(__name__)


def push_folder_to_hub(
    folder_path: Path, subfolder: str | None, repo_id: str, private: bool, token: str | None
) -> None:
    """
    Push a model folder to the huggingface hub, including model card.

    :param folder_path: The path to the folder.
    :param subfolder: The subfolder to push to.
        If None, the folder will be pushed to the root of the repo.
    :param repo_id: The repo name.
    :param private: Whether the repo is private.
    :param token: The huggingface token.
    """
    if not huggingface_hub.repo_exists(repo_id=repo_id, token=token):
        huggingface_hub.create_repo(repo_id, token=token, private=private)

    # Push model card and all model files to the Hugging Face hub
    huggingface_hub.upload_folder(repo_id=repo_id, folder_path=folder_path, token=token, path_in_repo=subfolder)

    logger.info(f"Pushed model to {repo_id}")


def maybe_get_cached_model_path(model_id: str) -> Path | None:
    """
    Gets the latest model path for a given identifier from the hugging face hub cache.

    Returns None if there is no cached model. In this case, the model will be downloaded.
    """
    # Early exit: a model_id must have a single '/'
    if model_id.count("/") != 1:
        return None
    # Make path object
    cache_dir = Path(HF_HUB_CACHE)
    # This is specific to how HF stores the files.
    normalized = model_id.replace("/", "--")
    repo_dir = cache_dir / f"models--{normalized}" / "snapshots"

    if not repo_dir.exists():
        return None

    # Find all directories.
    snapshots = [p for p in repo_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None

    # Get the latest directory by modification time.
    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
    return latest_snapshot
