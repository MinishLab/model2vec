from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar, cast

import huggingface_hub
import numpy as np
from huggingface_hub.errors import EntryNotFoundError
from safetensors.numpy import load_file, save_file

from model2vec.inference.evaluation import evaluate_single_or_multi_label
from model2vec.inference.mlp import Activation, Layer, MLPHead
from model2vec.model import PathLike, StaticModel
from model2vec.persistence import save_pretrained

_DEFAULT_HEAD_FILENAME = "head.safetensors"
_LEGACY_HEAD_FILENAME = "pipeline.skops"

LabelType = TypeVar("LabelType", list[str], list[list[str]])


class StaticModelPipeline:
    def __init__(self, model: StaticModel, head: MLPHead) -> None:
        """Create a pipeline with a StaticModel encoder."""
        self.model = model
        self.head = head
        self.classes_ = head.classes_

    @classmethod
    def from_pretrained(
        cls: type[StaticModelPipeline], path: PathLike, token: str | None = None
    ) -> StaticModelPipeline:
        """Load a StaticModel from a local path or huggingface hub path.

        NOTE: if you load a private model from the huggingface hub, you need to pass a token.

        NOTE: if the pipeline was saved by an older version of model2vec (a `pipeline.skops` file instead of
        `head.safetensors`), this falls back to loading it as a legacy pipeline and emits a warning. This requires
        `scikit-learn` and `skops` to be installed. See `convert_legacy_pipeline` to upgrade it to the current format.

        :param path: The path to the folder containing the pipeline, or a repository on the Hugging Face Hub
        :param token: The token to use to download the pipeline from the hub.
        :return: The loaded pipeline.
        """
        model, head = _load_pipeline(path, token)
        model.embedding = np.nan_to_num(model.embedding)

        return cls(model, head)

    def save_pretrained(self, path: str) -> None:
        """Save the model to a folder."""
        _save_pipeline(self, path)

    def push_to_hub(
        self, repo_id: str, subfolder: str | None = None, token: str | None = None, private: bool = False
    ) -> None:
        """Save a model to a folder, and then push that folder to the hf hub.

        :param repo_id: The id of the repository to push to.
        :param subfolder: The subfolder to push to.
        :param token: The token to use to push to the hub.
        :param private: Whether the repository should be private.
        """
        from model2vec.persistence import push_folder_to_hub

        with TemporaryDirectory() as temp_dir:
            _save_pipeline(self, temp_dir)
            push_folder_to_hub(Path(temp_dir), subfolder, repo_id, private, token)

    def _encode_and_coerce_to_2d(
        self,
        X: Sequence[str],
        show_progress_bar: bool,
        max_length: int | None,
        batch_size: int,
        use_multiprocessing: bool,
        multiprocessing_threshold: int,
    ) -> np.ndarray:
        """Encode the instances and coerce the output to a matrix."""
        encoded = self.model.encode(
            X,
            show_progress_bar=show_progress_bar,
            max_length=max_length,
            batch_size=batch_size,
            use_multiprocessing=use_multiprocessing,
            multiprocessing_threshold=multiprocessing_threshold,
        )
        if np.ndim(encoded) == 1:
            encoded = encoded[None, :]

        return encoded

    def predict(
        self,
        X: Sequence[str],
        show_progress_bar: bool = False,
        max_length: int | None = 512,
        batch_size: int = 1024,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict the labels of the input.

        :param X: The input data to predict. Can be a list of strings or a single string.
        :param show_progress_bar: Whether to display a progress bar during prediction. Defaults to False.
        :param max_length: The maximum length of the input sequences. Defaults to 512.
        :param batch_size: The batch size for prediction. Defaults to 1024.
        :param use_multiprocessing: Whether to use multiprocessing for encoding. Defaults to True.
        :param multiprocessing_threshold: The threshold for the number of samples to use multiprocessing. Defaults to 10,000.
        :param threshold: The threshold for multilabel classification. Defaults to 0.5. Ignored if not multilabel.
        :return: The predicted labels or probabilities.
        """
        encoded = self._encode_and_coerce_to_2d(
            X,
            show_progress_bar=show_progress_bar,
            max_length=max_length,
            batch_size=batch_size,
            use_multiprocessing=use_multiprocessing,
            multiprocessing_threshold=multiprocessing_threshold,
        )

        if self.head.activation == Activation.IDENTITY:
            return self.head.predict_regression(encoded)

        if self.head.activation == Activation.SIGMOID:
            assert self.classes_ is not None
            proba = self.head.predict_proba(encoded)
            out_labels = [self.classes_[vector > threshold] for vector in proba]
            return np.asarray(out_labels, dtype=object)

        assert self.classes_ is not None
        return self.classes_[self.head.predict_index(encoded)]

    def predict_proba(
        self,
        X: Sequence[str],
        show_progress_bar: bool = False,
        max_length: int | None = 512,
        batch_size: int = 1024,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
    ) -> np.ndarray:
        """Predict the labels of the input.

        :param X: The input data to predict. Can be a list of strings or a single string.
        :param show_progress_bar: Whether to display a progress bar during prediction. Defaults to False.
        :param max_length: The maximum length of the input sequences. Defaults to 512.
        :param batch_size: The batch size for prediction. Defaults to 1024.
        :param use_multiprocessing: Whether to use multiprocessing for encoding. Defaults to True.
        :param multiprocessing_threshold: The threshold for the number of samples to use multiprocessing. Defaults to 10,000.
        :return: The predicted labels or probabilities.
        :raises ValueError: If the classifier type is projector.
        """
        if self.head.activation == Activation.IDENTITY:
            raise ValueError("You are using evaluate on a projector model. This is not supported.")
        encoded = self._encode_and_coerce_to_2d(
            X,
            show_progress_bar=show_progress_bar,
            max_length=max_length,
            batch_size=batch_size,
            use_multiprocessing=use_multiprocessing,
            multiprocessing_threshold=multiprocessing_threshold,
        )

        return self.head.predict_proba(encoded)

    def evaluate(
        self, X: Sequence[str], y: LabelType, batch_size: int = 1024, threshold: float = 0.5
    ) -> dict[str, dict[str, float]]:
        """Evaluate the classifier on a given dataset using a classification report.

        :param X: The texts to predict on.
        :param y: The ground truth labels.
        :param batch_size: The batch size.
        :param threshold: The threshold for multilabel classification.
        :return: A classification report, as a dictionary.
        :raises ValueError: If the classifier type is projector.
        """
        if self.head.activation == Activation.IDENTITY:
            raise ValueError("You are using evaluate on a projector model. This is not supported.")
        predictions = self.predict(X, show_progress_bar=True, batch_size=batch_size, threshold=threshold)
        return evaluate_single_or_multi_label(predictions=predictions, y=y)


def _load_pipeline(folder_or_repo_path: PathLike, token: str | None = None) -> tuple[StaticModel, MLPHead]:
    """Load a model and its head.

    This assumes the following files are present in the repo:
    - `head.safetensors`: The weights of the head.
    - `config.json`: The configuration of the model, including the head's metadata.
    - `model.safetensors`: The weights of the model.
    - `tokenizer.json`: The tokenizer of the model.

    :param folder_or_repo_path: The path to the folder containing the pipeline.
    :param token: The token to use to download the pipeline from the hub. If this is None, you will only
        be able to load the pipeline from a local folder, public repository, or a repository that you have access to
        because you are logged in.
    :return: The encoder model and the loaded head
    :raises FileNotFoundError: If neither the head file nor a legacy pipeline file exist in the folder.
    """
    folder_or_repo_path = Path(folder_or_repo_path)
    head_path: str | Path
    if folder_or_repo_path.exists():
        head_path = folder_or_repo_path / _DEFAULT_HEAD_FILENAME
        if not head_path.exists():
            if not (folder_or_repo_path / _LEGACY_HEAD_FILENAME).exists():
                raise FileNotFoundError(f"Head file does not exist in {folder_or_repo_path}")
            return _load_legacy_pipeline(folder_or_repo_path, token)
    else:
        try:
            head_path = huggingface_hub.hf_hub_download(
                folder_or_repo_path.as_posix(), _DEFAULT_HEAD_FILENAME, token=token
            )
        except EntryNotFoundError:
            return _load_legacy_pipeline(folder_or_repo_path, token)

    model = StaticModel.from_pretrained(folder_or_repo_path)

    head_config = cast(dict[str, Any], model.config.get("head_config", {}))
    activation = Activation(head_config.get("activation", Activation.IDENTITY.value))
    n_layers = head_config.get("n_layers", 0)
    classes = head_config.get("classes")

    tensors = load_file(head_path)
    layers = [
        Layer(weight=tensors[f"head.{index}.weight"], bias=tensors[f"head.{index}.bias"]) for index in range(n_layers)
    ]

    head = MLPHead(
        layers=layers,
        activation=activation,
        classes=np.asarray(classes) if classes is not None else None,
    )

    return model, head


def _load_legacy_pipeline(folder_or_repo_path: PathLike, token: str | None) -> tuple[StaticModel, MLPHead]:
    """Load a model and its head from a legacy (scikit-learn/skops based) pipeline."""
    _legacy_warning = (
        f"No `{_DEFAULT_HEAD_FILENAME}` found for {folder_or_repo_path}; falling back to the legacy "
        f"`{_LEGACY_HEAD_FILENAME}` format. Save this pipeline with `save_pretrained` to upgrade it."
    )
    warnings.warn(_legacy_warning, stacklevel=3)
    converted = convert_legacy_pipeline(folder_or_repo_path, token=token)
    return converted.model, converted.head


def convert_legacy_pipeline(
    path: PathLike, token: str | None = None, trust_remote_code: bool = False
) -> StaticModelPipeline:
    """Convert an old-style (scikit-learn/skops based) `StaticModelPipeline` to the current, safetensors-based format.

    This requires `scikit-learn` and `skops` to be installed.

    :param path: The path to a local folder, or a repository on the Hugging Face Hub, containing a legacy pipeline
        saved as a `pipeline.skops` file.
    :param token: The token to use to download the pipeline from the hub.
    :param trust_remote_code: Whether to trust the remote code in the skops file. If this is False, we will only
        load components coming from `sklearn`.
    :return: A new-style `StaticModelPipeline`, which can be persisted with `save_pretrained` or `push_to_hub`.
    :raises ImportError: If `scikit-learn` and `skops` are not installed.
    :raises FileNotFoundError: If the pipeline file does not exist in the folder.
    :raises ValueError: If an untrusted type is found in the pipeline, or the head is not an MLP with a supported
        hidden activation.
    """
    import re

    try:
        import skops.io
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError("Converting a legacy pipeline requires `scikit-learn` and `skops`. ") from exc

    folder_or_repo_path = Path(path)
    legacy_head_path: str | Path
    if folder_or_repo_path.exists():
        legacy_head_path = folder_or_repo_path / _LEGACY_HEAD_FILENAME
        if not legacy_head_path.exists():
            raise FileNotFoundError(f"Pipeline file does not exist in {folder_or_repo_path}")
    else:
        legacy_head_path = huggingface_hub.hf_hub_download(
            folder_or_repo_path.as_posix(), _LEGACY_HEAD_FILENAME, token=token
        )

    model = StaticModel.from_pretrained(folder_or_repo_path)
    model.embedding = np.nan_to_num(model.embedding)

    untrusted_types = skops.io.get_untrusted_types(file=legacy_head_path)
    if not trust_remote_code:
        trusted_pattern = re.compile(r"sklearn\..+")
        for untrusted_type in untrusted_types:
            if not trusted_pattern.match(untrusted_type):
                raise ValueError(f"Untrusted type {untrusted_type}.")
    legacy_pipeline = cast(Pipeline, skops.io.load(legacy_head_path, trusted=untrusted_types))
    legacy_head = legacy_pipeline[-1]

    if isinstance(legacy_head, MLPRegressor):
        activation = Activation.IDENTITY
        classes = None
    elif isinstance(legacy_head, MLPClassifier):
        activation = Activation.SIGMOID if legacy_head.out_activation_ == "logistic" else Activation.SOFTMAX
        classes = np.asarray(legacy_head.classes_)
    else:
        raise ValueError(f"Unsupported legacy head type: {type(legacy_head)}. Expected MLPClassifier or MLPRegressor.")

    if legacy_head.activation != "relu":
        raise ValueError(f"Unsupported hidden activation: {legacy_head.activation}. Only `relu` is supported.")

    layers = [
        Layer(weight=coef.T, bias=intercept) for coef, intercept in zip(legacy_head.coefs_, legacy_head.intercepts_)
    ]
    head = MLPHead(layers=layers, activation=activation, classes=classes)

    return StaticModelPipeline(model, head)


def _save_pipeline(pipeline: StaticModelPipeline, folder_path: str | Path) -> None:
    """Save a pipeline to a folder.

    :param pipeline: The pipeline to save.
    :param folder_path: The path to the folder to save the pipeline to.
    """
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    head = pipeline.head
    tensors: dict[str, np.ndarray] = {}
    for index, layer in enumerate(head.layers):
        tensors[f"head.{index}.weight"] = layer.weight
        tensors[f"head.{index}.bias"] = layer.bias
    save_file(tensors, folder_path / _DEFAULT_HEAD_FILENAME)

    model = pipeline.model
    config = dict(model.config)
    config["head_config"] = {
        "n_layers": len(head.layers),
        "activation": head.activation.value,
        "classes": np.asarray(pipeline.classes_).tolist() if pipeline.classes_ is not None else None,
    }

    base_model_name = model.base_model_name
    if isinstance(base_model_name, list) and base_model_name:
        name = base_model_name[0]
    elif isinstance(base_model_name, str):
        name = base_model_name
    else:
        name = "unknown"

    save_pretrained(
        folder_path=folder_path,
        embeddings=model.embedding,
        tokenizer=model.tokenizer,
        config=config,
        base_model_name=name,
        language=model.language,
        weights=model.weights,
        mapping=model.token_mapping,
        template_path="classifier_template.md",
    )
