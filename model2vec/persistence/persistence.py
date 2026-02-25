from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import numpy as np
import safetensors
from safetensors.numpy import save_file
from tokenizers import Tokenizer

from model2vec.modelcards import create_model_card as make_model_card
from model2vec.modelcards import get_metadata_from_readme
from model2vec.persistence.datamodels import FOLDER_LAYOUTS, Layout
from model2vec.persistence.hf import maybe_get_cached_model_path
from model2vec.utils import SafeOpenProtocol

logger = logging.getLogger(__name__)


def save_pretrained(
    folder_path: Path,
    embeddings: np.ndarray,
    tokenizer: Tokenizer,
    config: dict[str, Any],
    create_model_card: bool = True,
    subfolder: str | None = None,
    weights: np.ndarray | None = None,
    mapping: np.ndarray | None = None,
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
    :param weights: The weights of the model. If None, no weights are saved.
    :param mapping: The token mapping of the model. If None, there is no token mapping.
    :param **kwargs: Any additional arguments.
    """
    folder_path = folder_path / subfolder if subfolder else folder_path
    folder_path.mkdir(exist_ok=True, parents=True)

    model_weights = {"embeddings": embeddings}
    if weights is not None:
        model_weights["weights"] = weights
    if mapping is not None:
        model_weights["mapping"] = mapping

    save_file(model_weights, folder_path / "model.safetensors")
    tokenizer.save(str(folder_path / "tokenizer.json"), pretty=False)

    # Create a copy of config and add dtype and vocab quantization
    cfg = dict(config)
    cfg["embedding_dtype"] = np.dtype(embeddings.dtype).name
    if mapping is not None:
        cfg["vocabulary_quantization"] = int(embeddings.shape[0])
    else:
        cfg.pop("vocabulary_quantization", None)
    json.dump(cfg, open(folder_path / "config.json", "w"), indent=4)

    # Create modules.json
    modules = [{"idx": 0, "name": "0", "path": ".", "type": "sentence_transformers.models.StaticEmbedding"}]
    if cfg.get("normalize"):
        # If normalize=True, add sentence_transformers.models.Normalize
        modules.append({"idx": 1, "name": "1", "path": "1_Normalize", "type": "sentence_transformers.models.Normalize"})
    json.dump(modules, open(folder_path / "modules.json", "w"), indent=4)

    logger.info(f"Saved model to {folder_path}")

    # Optionally create the model card
    if create_model_card:
        make_model_card(folder_path, **kwargs)


def load_pretrained(
    folder_or_repo_path: str | Path,
    subfolder: str | None,
    token: str | None,
    force_download: bool,
) -> tuple[np.ndarray, Tokenizer, dict[str, Any], dict[str, Any], np.ndarray | None, np.ndarray | None]:
    """
    Loads a pretrained model from a folder.

    :param folder_or_repo_path: The folder or repo path to load from.
        - If this is a local path, we will load from the local path.
        - If the local path is not found, we will attempt to load from the huggingface hub.
    :param subfolder: The subfolder to load from.
    :param token: The huggingface token to use.
    :param force_download: Whether to force the download of the model. If False, the model is only downloaded if it is not
        already present in the cache.
    :return: The embeddings, tokenizer, config, metadata, weights and mapping.

    """
    folder_or_repo_path = Path(folder_or_repo_path)

    # We resolve a folder or repo path to an actual local folder.
    folder = _resolve_folder(folder_or_repo_path=folder_or_repo_path, token=token, force_download=force_download)

    if subfolder:
        folder = folder / subfolder

    selected_layout = _get_paths(folder)
    readme_path = folder / "README.md"

    opened_tensor_file = cast(SafeOpenProtocol, safetensors.safe_open(selected_layout.embeddings, framework="numpy"))
    embedding_name = "embedding.weight" if selected_layout.is_sentence_transformers else "embeddings"
    # Actual loading
    embeddings = opened_tensor_file.get_tensor(embedding_name)
    try:
        weights = opened_tensor_file.get_tensor("weights")
    except Exception:
        # Bare except because safetensors does not export its own errors.
        weights = None
    try:
        mapping = opened_tensor_file.get_tensor("mapping")
    except Exception:
        mapping = None

    if readme_path.exists():
        metadata = get_metadata_from_readme(readme_path)
    else:
        metadata = {}

    tokenizer: Tokenizer = Tokenizer.from_file(str(selected_layout.tokenizer))
    config = json.load(open(selected_layout.config))

    return embeddings, tokenizer, config, metadata, weights, mapping


def _resolve_folder(folder_or_repo_path: Path, token: str | None, force_download: bool) -> Path:
    """Resolve a folder locally or from hugging face hub."""
    if folder_or_repo_path.exists():
        return folder_or_repo_path
    # We now know we're dealing with either an invalid path, or
    # a HF model ID.
    if not force_download:
        if folder := maybe_get_cached_model_path(str(folder_or_repo_path)):
            return folder

    folder = Path(huggingface_hub.snapshot_download(str(folder_or_repo_path), repo_type="model", token=token))

    return folder


def _get_paths(folder: Path) -> Layout:
    """Get all paths by trying out multiple layouts."""
    for layout in FOLDER_LAYOUTS:
        layout = layout.with_parent(folder)
        if layout.is_valid():
            return layout

    raise ValueError(
        f"Could not find expected model files in {folder}. "
        "Tried model2vec, sentence-transformers, and 0_StaticEmbedding layouts."
    )
