import logging
from pathlib import Path
from typing import Any

from huggingface_hub import ModelCard, ModelCardData

logger = logging.getLogger(__name__)


def create_model_card(
    folder_path: Path,
    base_model_name: str = "unknown",
    license: str = "mit",
    language: list[str] | None = None,
    model_name: str | None = None,
    template_path: str = "model_card_template.md",
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


def get_metadata_from_readme(readme_path: Path) -> dict[str, Any]:
    """Get metadata from a README file."""
    if not readme_path.exists():
        logger.info(f"README file not found in {readme_path}. No model card loaded.")
        return {}
    model_card = ModelCard.load(readme_path)
    data: dict[str, Any] = model_card.data.to_dict()
    if not data:
        logger.info("File README.md exists, but was empty. No model card loaded.")
    return data
