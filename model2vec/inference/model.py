from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, TypeVar

import huggingface_hub
import numpy as np
import skops.io
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from model2vec.hf_utils import _create_model_card
from model2vec.model import PathLike, StaticModel

_DEFAULT_TRUST_PATTERN = re.compile(r"sklearn\..+")
_DEFAULT_MODEL_FILENAME = "pipeline.skops"

LabelType = TypeVar("LabelType", list[str], list[list[str]])


class StaticModelPipeline:
    def __init__(self, model: StaticModel, head: Pipeline) -> None:
        """Create a pipeline with a StaticModel encoder."""
        self.model = model
        self.head = head
        classifier = self.head[-1]
        # Check if the classifier is a multilabel classifier.
        # NOTE: this doesn't look robust, but it is.
        # Different classifiers, such as OVR wrappers, support multilabel output natively, so we
        # can just use predict.
        self.multilabel = False
        if isinstance(classifier, MLPClassifier):
            if classifier.out_activation_ == "logistic":
                self.multilabel = True

    @property
    def classes_(self) -> np.ndarray:
        """The classes of the classifier."""
        return self.head.classes_

    @classmethod
    def from_pretrained(
        cls: type[StaticModelPipeline], path: PathLike, token: str | None = None, trust_remote_code: bool = False
    ) -> StaticModelPipeline:
        """
        Load a StaticModel from a local path or huggingface hub path.

        NOTE: if you load a private model from the huggingface hub, you need to pass a token.

        :param path: The path to the folder containing the pipeline, or a repository on the Hugging Face Hub
        :param token: The token to use to download the pipeline from the hub.
        :param trust_remote_code: Whether to trust the remote code. If this is False, we will only load components coming from `sklearn`.
        :return: The loaded pipeline.
        """
        model, head = _load_pipeline(path, token, trust_remote_code)
        model.embedding = np.nan_to_num(model.embedding)

        return cls(model, head)

    def save_pretrained(self, path: str) -> None:
        """Save the model to a folder."""
        save_pipeline(self, path)

    def push_to_hub(
        self, repo_id: str, subfolder: str | None = None, token: str | None = None, private: bool = False
    ) -> None:
        """
        Save a model to a folder, and then push that folder to the hf hub.

        :param repo_id: The id of the repository to push to.
        :param subfolder: The subfolder to push to.
        :param token: The token to use to push to the hub.
        :param private: Whether the repository should be private.
        """
        from model2vec.hf_utils import push_folder_to_hub

        with TemporaryDirectory() as temp_dir:
            save_pipeline(self, temp_dir)
            self.model.save_pretrained(temp_dir)
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
        """
        Predict the labels of the input.

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

        if self.multilabel:
            out_labels = []
            proba = self.head.predict_proba(encoded)
            for vector in proba:
                out_labels.append(self.classes_[vector > threshold])
            return np.asarray(out_labels, dtype=object)

        return self.head.predict(encoded)

    def predict_proba(
        self,
        X: Sequence[str],
        show_progress_bar: bool = False,
        max_length: int | None = 512,
        batch_size: int = 1024,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
    ) -> np.ndarray:
        """
        Predict the labels of the input.

        :param X: The input data to predict. Can be a list of strings or a single string.
        :param show_progress_bar: Whether to display a progress bar during prediction. Defaults to False.
        :param max_length: The maximum length of the input sequences. Defaults to 512.
        :param batch_size: The batch size for prediction. Defaults to 1024.
        :param use_multiprocessing: Whether to use multiprocessing for encoding. Defaults to True.
        :param multiprocessing_threshold: The threshold for the number of samples to use multiprocessing. Defaults to 10,000.
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

        return self.head.predict_proba(encoded)

    def evaluate(
        self, X: Sequence[str], y: LabelType, batch_size: int = 1024, threshold: float = 0.5, output_dict: bool = False
    ) -> str | dict[str, dict[str, float]]:
        """
        Evaluate the classifier on a given dataset using scikit-learn's classification report.

        :param X: The texts to predict on.
        :param y: The ground truth labels.
        :param batch_size: The batch size.
        :param threshold: The threshold for multilabel classification.
        :param output_dict: Whether to output the classification report as a dictionary.
        :return: A classification report.
        """
        predictions = self.predict(X, show_progress_bar=True, batch_size=batch_size, threshold=threshold)
        report = evaluate_single_or_multi_label(predictions=predictions, y=y, output_dict=output_dict)

        return report


def _load_pipeline(
    folder_or_repo_path: PathLike, token: str | None = None, trust_remote_code: bool = False
) -> tuple[StaticModel, Pipeline]:
    """
    Load a model and an sklearn pipeline.

    This assumes the following files are present in the repo:
    - `pipeline.skops`: The head of the pipeline.
    - `config.json`: The configuration of the model.
    - `model.safetensors`: The weights of the model.
    - `tokenizer.json`: The tokenizer of the model.

    :param folder_or_repo_path: The path to the folder containing the pipeline.
    :param token: The token to use to download the pipeline from the hub. If this is None, you will only
        be able to load the pipeline from a local folder, public repository, or a repository that you have access to
        because you are logged in.
    :param trust_remote_code: Whether to trust the remote code. If this is False,
        we will only load components coming from `sklearn`. If this is True, we will load all components.
        If you set this to True, you are responsible for whatever happens.
    :return: The encoder model and the loaded head
    :raises FileNotFoundError: If the pipeline file does not exist in the folder.
    :raises ValueError: If an untrusted type is found in the pipeline, and `trust_remote_code` is False.
    """
    folder_or_repo_path = Path(folder_or_repo_path)
    model_filename = _DEFAULT_MODEL_FILENAME
    head_pipeline_path: str | Path
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


def save_pipeline(pipeline: StaticModelPipeline, folder_path: str | Path) -> None:
    """
    Save a pipeline to a folder.

    :param pipeline: The pipeline to save.
    :param folder_path: The path to the folder to save the pipeline to.
    """
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    model_filename = _DEFAULT_MODEL_FILENAME
    head_pipeline_path = folder_path / model_filename
    skops.io.dump(pipeline.head, head_pipeline_path)
    pipeline.model.save_pretrained(folder_path)
    base_model_name = pipeline.model.base_model_name
    if isinstance(base_model_name, list) and base_model_name:
        name = base_model_name[0]
    elif isinstance(base_model_name, str):
        name = base_model_name
    else:
        name = "unknown"
    _create_model_card(
        folder_path,
        base_model_name=name,
        language=pipeline.model.language,
        template_path="modelcards/classifier_template.md",
    )


def _is_multi_label_shaped(y: LabelType) -> bool:
    """Check if the labels are in a multi-label shape."""
    return isinstance(y, (list, tuple)) and len(y) > 0 and isinstance(y[0], (list, tuple, set))


def evaluate_single_or_multi_label(
    predictions: np.ndarray,
    y: LabelType,
    output_dict: bool = False,
) -> str | dict[str, dict[str, float]]:
    """
    Evaluate the classifier on a given dataset using scikit-learn's classification report.

    :param predictions: The predictions.
    :param y: The ground truth labels.
    :param output_dict: Whether to output the classification report as a dictionary.
    :return: A classification report.
    """
    if _is_multi_label_shaped(y):
        classes = sorted(set([label for labels in y for label in labels]))
        mlb = MultiLabelBinarizer(classes=classes)
        y = mlb.fit_transform(y)
        predictions = mlb.transform(predictions)
    elif isinstance(y[0], (str, int)):
        classes = sorted(set(y))

    report = classification_report(
        y,
        predictions,
        output_dict=output_dict,
        zero_division=0,
    )

    return report
