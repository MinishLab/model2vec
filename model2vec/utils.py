# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
from typing import Any, Protocol, cast

import click
import huggingface_hub
import numpy as np
import safetensors
from huggingface_hub import ModelCard, ModelCardData
from rich.logging import RichHandler
from safetensors.numpy import save_file
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class SafeOpenProtocol(Protocol):
    """Protocol to fix safetensors safe open."""

    def get_tensor(self, key: str) -> np.ndarray:
        """Get a tensor."""
        ...


def setup_logging() -> None:
    """Simple logging setup."""
    logging.basicConfig(
        level="INFO",
        format="%(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
    )


def save_pretrained(
    folder_path: Path,
    embeddings: np.ndarray,
    tokenizer: Tokenizer,
    config: dict[str, Any],
    create_model_card: bool = True,
    **kwargs: Any,
) -> None:
    """
    Save a model to a folder.

    :param folder_path: The path to the folder.
    :param embeddings: The embeddings.
    :param tokenizer: The tokenizer.
    :param config: A metadata config.
    :param create_model_card: Whether to create a model card.
    :param **kwargs: Any additional arguments.
    """
    folder_path.mkdir(exist_ok=True, parents=True)
    save_file({"embeddings": embeddings}, folder_path / "embeddings.safetensors")
    tokenizer.save(str(folder_path / "tokenizer.json"))
    json.dump(config, open(folder_path / "config.json", "w"))

    logger.info(f"Saved model to {folder_path}")

    # Optionally create the model card
    if create_model_card:
        _create_model_card(folder_path, **kwargs)


def _create_model_card(folder_path: Path, **kwargs: Any) -> None:
    """
    Create a model card and store it in the specified path.

    Args:
    ----
        folder_path (str): The path where the model card will be stored.
        config (dict): A dictionary containing essential settings for running the model.
        **kwargs: Additional metadata for the model card (e.g., model_name, base_model, etc.).

    """
    folder_path = Path(folder_path)
    model_name = folder_path.name

    # Generate the model card content
    model_card = _generate_model_card(model_name, **kwargs)

    # Save the model card as README.md
    with open(folder_path / "README.md", "w", encoding="utf8") as fOut:
        fOut.write(model_card)


def _generate_model_card(model_name: str, **kwargs: Any) -> str:
    """
    Generate a model card using a string template.

    :param model_name: The name of the model.
    :param **kwargs: Additional metadata for the model card (e.g. language, base_model, etc.).
    :return: The content of the model card in markdown format.
    """
    # Retrieve metadata from kwargs
    base_model = kwargs.get("base_model_name", "unknown")
    language = kwargs.get("language", "unknown")
    license = kwargs.get("license", "MIT")

    # Define the content for the model card
    model_card_content = f"""

    # {model_name} Model Card

    Model2Vec distills a Sentence Transformer into a small, static model.
    This model is ideal for applications requiring fast, lightweight embeddings.

    ---
    model_name: {model_name}
    base_model: {base_model}
    library: 'https://github.com/MinishLab/model2vec'
    language: {language}
    license: {license}
    ---

    ## Installation

    Install model2vec using pip:
    ```
    pip install model2vec
    ```

    ## Usage
    A StaticModel can be loaded using the `from_pretrained` method:
    ```python
    from model2vec import StaticModel
    model = StaticModel.from_pretrained("minishlab/M2V_base_output")
    embeddings = model.encode(["Example sentence"])
    ```

    Alternatively, you can distill your own model using the `distill` method:
    ```python
    from model2vec.distill import distill

    # Choose a Sentence Transformer model
    model_name = "BAAI/bge-base-en-v1.5"

    # Distill the model
    m2v_model = distill(model_name=model_name, pca_dims=256)

    # Save the model
    m2v_model.save_pretrained("m2v_model")
    ```

    ## How it works

    Model2vec creates a small, fast, and powerful model that outperforms other static embedding models by a large margin on all tasks we could find, while being much faster to create than traditional static embedding models such as GloVe. Best of all, you don't need any data to distill a model using Model2Vec.

    It works by passing a vocabulary through a sentence transformer model, then reducing the dimensionality of the resulting embeddings using PCA, and finally weighting the embeddings using zipf weighting. During inference, we simply take the mean of all token embeddings occurring in a sentence.

    ## Citation

    Please cite the Model2Vec repository if you use this model in your work.

    ## Additional Resources

    - [Model2Vec Repo](https://github.com/MinishLab/model2vec)
    - [Model2Vec Results](https://github.com/MinishLab/model2vec?tab=readme-ov-file#results)
    - [Model2Vec Tutorials](https://github.com/MinishLab/model2vec/tree/main/tutorials)

    ## Model Authors

    Model2Vec was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of Stephan Tulkens and Thomas van Dongen.
    """

    return model_card_content


def load_pretrained(
    folder_or_repo_path: str | Path, token: str | None = None
) -> tuple[np.ndarray, Tokenizer, dict[str, Any]]:
    """
    Loads a pretrained model from a folder.

    :param folder_or_repo_path: The folder or repo path to load from.
        - If this is a local path, we will load from the local path.
        - If the local path is not found, we will attempt to load from the huggingface hub.
    :param token: The huggingface token to use.
    :raises: FileNotFoundError if the folder exists, but the file does not exist locally.
    :return: The embeddings, tokenizer, and config.

    """
    folder_or_repo_path = Path(folder_or_repo_path)
    if folder_or_repo_path.exists():
        embeddings_path = folder_or_repo_path / "embeddings.safetensors"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file does not exist in {folder_or_repo_path}")

        config_path = folder_or_repo_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist in {folder_or_repo_path}")

        tokenizer_path = folder_or_repo_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file does not exist in {folder_or_repo_path}")

    else:
        logger.info("Folder does not exist locally, attempting to use huggingface hub.")
        embeddings_path = huggingface_hub.hf_hub_download(
            str(folder_or_repo_path), "embeddings.safetensors", token=token
        )
        config_path = huggingface_hub.hf_hub_download(str(folder_or_repo_path), "config.json", token=token)
        tokenizer_path = huggingface_hub.hf_hub_download(str(folder_or_repo_path), "tokenizer.json", token=token)

    opened_tensor_file = cast(SafeOpenProtocol, safetensors.safe_open(embeddings_path, framework="numpy"))
    embeddings = opened_tensor_file.get_tensor("embeddings")

    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = json.load(open(config_path))

    if len(tokenizer.get_vocab()) != len(embeddings):
        logger.warning(
            f"Number of tokens does not match number of embeddings: `{len(tokenizer.get_vocab())}` vs `{len(embeddings)}`"
        )

    return embeddings, tokenizer, config


def push_folder_to_hub(folder_path: Path, repo_id: str, token: str | None) -> None:
    """
    Push a model folder to the huggingface hub.

    :param folder_path: The path to the folder.
    :param repo_id: The repo name.
    :param token: The huggingface token.
    """
    if not huggingface_hub.repo_exists(repo_id=repo_id, token=token):
        huggingface_hub.create_repo(repo_id, token=token)
    huggingface_hub.upload_folder(repo_id=repo_id, folder_path=folder_path, token=token)
    logger.info(f"Pushed model to {repo_id}")
