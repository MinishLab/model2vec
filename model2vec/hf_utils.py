from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import numpy as np
import safetensors
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors.numpy import save_file
from tokenizers import Tokenizer

from model2vec.utils import SafeOpenProtocol

logger = logging.getLogger(__name__)


def save_pretrained(
    folder_path: Path,
    embeddings: np.ndarray,
    tokenizer: Tokenizer,
    config: dict[str, Any],
    create_model_card: bool = True,
    subfolder: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Save a model to a folder.

    :param folder_path: The path to the folder.
    :param embeddings: The embeddings.
    :param tokenizer: The tokenizer.
    :param config: A metadata config.
    :param create_model_card: Whether to create a model card.
    :param subfolder: The subfolder to save the model in.
    :param **kwargs: Any additional arguments.
    """
    folder_path = folder_path / subfolder if subfolder else folder_path
    folder_path.mkdir(exist_ok=True, parents=True)
    save_file({"embeddings": embeddings}, folder_path / "model.safetensors")
    tokenizer.save(str(folder_path / "tokenizer.json"), pretty=False)
    json.dump(config, open(folder_path / "config.json", "w"), indent=4)

    # Create modules.json
    modules = [{"idx": 0, "name": "0", "path": ".", "type": "sentence_transformers.models.StaticEmbedding"}]
    if config.get("normalize"):
        # If normalize=True, add sentence_transformers.models.Normalize
        modules.append({"idx": 1, "name": "1", "path": "1_Normalize", "type": "sentence_transformers.models.Normalize"})
    json.dump(modules, open(folder_path / "modules.json", "w"), indent=4)

    logger.info(f"Saved model to {folder_path}")

    # Optionally create the model card
    if create_model_card:
        _create_model_card(folder_path, **kwargs)


def _create_model_card(
    folder_path: Path,
    base_model_name: str = "unknown",
    license: str = "mit",
    language: list[str] | None = None,
    model_name: str | None = None,
    template_path: str = "modelcards/model_card_template.md",
    **kwargs: Any,
) -> None:
    """
    Create a model card and store it in the specified path.

    :param folder_path: The path where the model card will be stored.
    :param base_model_name: The name of the base model.
    :param license: The license to use.
    :param language: The language of the model.
    :param model_name: The name of the model to use in the Model Card.
    :param template_path: The path to the template.
    :param **kwargs: Additional metadata for the model card (e.g., model_name, base_model, etc.).
    """
    folder_path = Path(folder_path)
    model_name = model_name or folder_path.name
    full_path = Path(__file__).parent / template_path

    model_card_data = ModelCardData(
        model_name=model_name,
        base_model=base_model_name,
        license=license,
        language=language,
        tags=["embeddings", "static-embeddings", "sentence-transformers"],
        library_name="model2vec",
        **kwargs,
    )
    model_card = ModelCard.from_template(model_card_data, template_path=str(full_path))
    model_card.save(folder_path / "README.md")


def load_pretrained(
    folder_or_repo_path: str | Path,
    subfolder: str | None = None,
    token: str | None = None,
    from_sentence_transformers: bool = False,
    skip_metadata: bool = False,
) -> tuple[np.ndarray, Tokenizer, dict[str, Any], dict[str, Any]]:
    """
    Loads a pretrained model from a folder.

    :param folder_or_repo_path: The folder or repo path to load from.
        - If this is a local path, we will load from the local path.
        - If the local path is not found, we will attempt to load from the huggingface hub.
    :param subfolder: The subfolder to load from.
    :param token: The huggingface token to use.
    :param from_sentence_transformers: Whether to load the model from a sentence transformers model.
    :param skip_metadata: Whether to skip loading metadata. This is useful if you don't need the metadata.
    :raises: FileNotFoundError if the folder exists, but the file does not exist locally.
    :return: The embeddings, tokenizer, config, and metadata.

    """
    if from_sentence_transformers:
        model_file = "0_StaticEmbedding/model.safetensors"
        tokenizer_file = "0_StaticEmbedding/tokenizer.json"
        config_name = "config_sentence_transformers.json"
    else:
        model_file = "model.safetensors"
        tokenizer_file = "tokenizer.json"
        config_name = "config.json"

    if cached_folder := _get_latest_model_path(str(folder_or_repo_path)):
        logger.info(f"Found cached model at {cached_folder}, loading from cache.")
        folder_or_repo_path = cached_folder
    else:
        logger.info(f"No cached model found for {folder_or_repo_path}, loading from local or hub.")
        folder_or_repo_path = Path(folder_or_repo_path)

    local_folder = folder_or_repo_path / subfolder if subfolder else folder_or_repo_path

    if local_folder.exists():
        embeddings_path = local_folder / model_file
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file does not exist in {local_folder}")

        config_path = local_folder / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist in {local_folder}")

        tokenizer_path = local_folder / tokenizer_file
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file does not exist in {local_folder}")

        readme_path = local_folder / "README.md"

    else:
        logger.info("Folder does not exist locally, attempting to use huggingface hub.")
        embeddings_path = Path(
            huggingface_hub.hf_hub_download(
                folder_or_repo_path.as_posix(), model_file, token=token, subfolder=subfolder
            )
        )
        readme_path = Path(
            huggingface_hub.hf_hub_download(
                folder_or_repo_path.as_posix(), "README.md", token=token, subfolder=subfolder
            )
        )

        config_path = Path(
            huggingface_hub.hf_hub_download(
                folder_or_repo_path.as_posix(), config_name, token=token, subfolder=subfolder
            )
        )
        tokenizer_path = Path(
            huggingface_hub.hf_hub_download(
                folder_or_repo_path.as_posix(), tokenizer_file, token=token, subfolder=subfolder
            )
        )

    opened_tensor_file = cast(SafeOpenProtocol, safetensors.safe_open(embeddings_path, framework="numpy"))
    embedding_key = "embedding.weight" if from_sentence_transformers else "embeddings"
    embeddings = opened_tensor_file.get_tensor(embedding_key)

    if not skip_metadata and readme_path.exists():
        metadata = _get_metadata_from_readme(readme_path)
    else:
        metadata = {}

    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = json.load(open(config_path))

    if len(tokenizer.get_vocab()) != len(embeddings):
        logger.warning(
            f"Number of tokens does not match number of embeddings: `{len(tokenizer.get_vocab())}` vs `{len(embeddings)}`"
        )

    return embeddings, tokenizer, config, metadata


def _get_metadata_from_readme(readme_path: Path) -> dict[str, Any]:
    """Get metadata from a README file."""
    if not readme_path.exists():
        logger.info(f"README file not found in {readme_path}. No model card loaded.")
        return {}
    model_card = ModelCard.load(readme_path)
    data: dict[str, Any] = model_card.data.to_dict()
    if not data:
        logger.info("File README.md exists, but was empty. No model card loaded.")
    return data


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


def _get_latest_model_path(model_id: str) -> Path | None:
    """
    Gets the latest model path for a given identifier from the hugging face hub cache.

    Returns None if there is no cached model. In this case, the model will be downloaded.
    """
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
