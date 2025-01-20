from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import huggingface_hub
import numpy as np
import skops.io
from sklearn.pipeline import Pipeline

from model2vec.model import PathLike, StaticModel

_DEFAULT_TRUST_PATTERN = re.compile("sklearn\..+")


class StaticModelPipeline:
    def __init__(self, model: StaticModel, head: Pipeline) -> None:
        """Create a pipeline in which the model is the encoder."""
        self.model = model
        self.head = head

    @classmethod
    def from_pretrained(
        cls: type[StaticModelPipeline], path: PathLike, token: str | None = None
    ) -> StaticModelPipeline:
        """Load the pipeline from the trained model."""
        model, head = _load_pipeline(path, token)

        return cls(model, head)

    def save_pretrained(self, path: str) -> None:
        """Push the pipeline to the hub."""
        save_pipeline(self, path)

    def push_to_hub(self, repo_id: str, token: str, private: bool = False) -> None:
        """Push the pipeline to the hub."""
        from model2vec.hf_utils import push_folder_to_hub

        with TemporaryDirectory() as temp_dir:
            save_pipeline(self, temp_dir)
            self.model.save_pretrained(temp_dir)
            push_folder_to_hub(Path(temp_dir), repo_id, private, token)

    def predict(self, X: list[str] | str) -> list[str]:
        """Predict the labels of the input."""
        encoded = self.model.encode(X)
        if np.ndim(encoded) == 1:
            encoded = encoded[None, :]

        return self.head.predict(encoded)

    def predict_proba(self, X: list[str] | str) -> np.ndarray:
        """Predict the probabilities of the labels of the input."""
        encoded = self.model.encode(X)
        if np.ndim(encoded) == 1:
            encoded = encoded[None, :]

        return self.head.predict_proba(encoded)


def _load_pipeline(
    folder_or_repo_path: PathLike, token: str | None = None, trust_remote_code: bool = False
) -> Pipeline:
    """Load the pipeline from the trained model."""
    folder_or_repo_path = Path(folder_or_repo_path)
    model_filename = "pipeline.skops"
    if folder_or_repo_path.exists():
        head_pipeline_path = folder_or_repo_path / model_filename
        if not head_pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file does not exist in {folder_or_repo_path}")
    else:
        head_pipeline_path = huggingface_hub.hf_hub_download(
            folder_or_repo_path.as_posix(), model_filename, token=token
        )

    model = StaticModel.from_pretrained(folder_or_repo_path)

    unknown_types = skops.io.get_untrusted_types(file=head_pipeline_path)
    # If the user does not trust remote code, we should check that the unknown types are trusted.
    # By default, we trust everything coming from scikit-learn.
    if not trust_remote_code:
        for t in unknown_types:
            if not _DEFAULT_TRUST_PATTERN.match(t):
                raise ValueError(f"Untrusted type {t}.")
    head = skops.io.load(head_pipeline_path, trusted=unknown_types)

    return model, head


def save_pipeline(pipeline: StaticModelPipeline, folder_or_repo_path: str | Path) -> None:
    """Saves a pipeline to a folder."""
    folder_or_repo_path = Path(folder_or_repo_path)
    folder_or_repo_path.mkdir(parents=True, exist_ok=True)
    model_filename = "pipeline.skops"
    head_pipeline_path = folder_or_repo_path / model_filename
    skops.io.dump(pipeline.head, head_pipeline_path)
    pipeline.model.save_pretrained(folder_or_repo_path)
