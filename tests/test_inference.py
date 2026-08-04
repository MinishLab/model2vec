import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest

from model2vec.inference.mlp import Activation, Layer, MLPHead
from model2vec.inference.model import StaticModelPipeline
from model2vec.model import StaticModel


def test_init_predict(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test successful init and predict with StaticModelPipeline."""
    target: list[str] | list[list[str]]
    if mock_inference_pipeline.head.activation == Activation.SIGMOID:
        assert mock_inference_pipeline.classes_ is not None
        if isinstance(mock_inference_pipeline.classes_[0], str):
            target = [["a", "b"]]
        else:
            target = [[0, 1]]  # type: ignore
    else:
        assert mock_inference_pipeline.classes_ is not None
        if isinstance(mock_inference_pipeline.classes_[0], str):
            target = ["b"]
        else:
            target = [1]  # type: ignore
    assert mock_inference_pipeline.predict("dog").tolist() == target
    assert mock_inference_pipeline.predict(["dog"]).tolist() == target


def test_init_predict_projector(mock_inference_pipeline_projector: StaticModelPipeline) -> None:
    """Test successful init and predict with StaticModelPipeline."""
    assert mock_inference_pipeline_projector.head.activation == Activation.IDENTITY
    assert mock_inference_pipeline_projector.classes_ is None
    with pytest.raises(ValueError):
        mock_inference_pipeline_projector.predict_proba(["dog"])
    with pytest.raises(ValueError):
        mock_inference_pipeline_projector.evaluate(["dog"], ["a"])

    prediction = mock_inference_pipeline_projector.predict(["dog"])
    assert prediction.shape == (1, 32)


def test_init_predict_proba(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test successful init and predict_proba with StaticModelPipeline."""
    assert mock_inference_pipeline.predict_proba("dog").argmax() == 1
    assert mock_inference_pipeline.predict_proba(["dog"]).argmax(1).tolist() == [1]


def test_init_evaluate(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test successful init and evaluate with StaticModelPipeline."""
    target: list[str] | list[list[str]]
    if mock_inference_pipeline.head.activation == Activation.SIGMOID:
        assert mock_inference_pipeline.classes_ is not None
        if isinstance(mock_inference_pipeline.classes_[0], str):
            target = [["a", "b"]]
        else:
            target = [[0, 1]]  # type: ignore
    else:
        assert mock_inference_pipeline.classes_ is not None
        if isinstance(mock_inference_pipeline.classes_[0], str):
            target = ["b"]
        else:
            target = [1]  # type: ignore
    mock_inference_pipeline.evaluate("dog", target)  # type: ignore


def test_roundtrip_save(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test saving and loading the pipeline."""
    with TemporaryDirectory() as temp_dir:
        mock_inference_pipeline.save_pretrained(temp_dir)
        loaded = StaticModelPipeline.from_pretrained(temp_dir)
        target: list[str] | list[list[str]]
        if mock_inference_pipeline.head.activation == Activation.SIGMOID:
            assert mock_inference_pipeline.classes_ is not None
            if isinstance(mock_inference_pipeline.classes_[0], str):
                target = [["a", "b"]]
            else:
                target = [[0, 1]]  # type: ignore
        else:
            assert mock_inference_pipeline.classes_ is not None
            if isinstance(mock_inference_pipeline.classes_[0], str):
                target = ["b"]
            else:
                target = [1]  # type: ignore
        assert loaded.predict("dog").tolist() == target
        assert loaded.predict(["dog"]).tolist() == target
        assert loaded.predict_proba("dog").argmax() == 1
        assert loaded.predict_proba(["dog"]).argmax(1).tolist() == [1]


def test_roundtrip_save_file_gone(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test saving and loading the pipeline."""
    with TemporaryDirectory() as temp_dir:
        mock_inference_pipeline.save_pretrained(temp_dir)
        # Remove the head file, so that it looks like it was never saved
        os.unlink(os.path.join(temp_dir, "head.safetensors"))
        with pytest.raises(FileNotFoundError):
            StaticModelPipeline.from_pretrained(temp_dir)


def test_mlp_head_predict_proba_identity() -> None:
    """Test that predict_proba returns the raw logits for an identity activation."""
    layer = Layer(weight=np.eye(3), bias=np.zeros(3))
    head = MLPHead(layers=[layer], activation=Activation.IDENTITY)

    X = np.array([[1.0, -2.0, 3.0]])
    proba = head.predict_proba(X)

    assert np.allclose(proba, X)


def test_load_pipeline_from_hub(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test that a repo id that isn't a local path is downloaded from the hub."""
    with TemporaryDirectory() as temp_dir:
        mock_inference_pipeline.save_pretrained(temp_dir)
        downloaded_model = StaticModel.from_pretrained(temp_dir)
        head_path = os.path.join(temp_dir, "head.safetensors")

        with (
            patch("model2vec.inference.model.huggingface_hub.hf_hub_download", return_value=head_path) as mock_download,
            patch("model2vec.inference.model.StaticModel.from_pretrained", return_value=downloaded_model),
        ):
            loaded = StaticModelPipeline.from_pretrained("fake/repo-id")

        mock_download.assert_called_once_with("fake/repo-id", "head.safetensors", token=None)
        assert loaded.predict(["dog"]).tolist() == mock_inference_pipeline.predict(["dog"]).tolist()


def test_push_to_hub(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test that push_to_hub saves the pipeline to a temp folder before pushing it to the hub."""
    captured: dict[str, object] = {}

    def _capture_upload(
        folder_path: Path, subfolder: str | None, repo_id: str, private: bool, token: str | None
    ) -> None:
        captured["files"] = sorted(p.name for p in folder_path.iterdir())
        captured["repo_id"] = repo_id
        captured["private"] = private

    with patch("model2vec.persistence.push_folder_to_hub", side_effect=_capture_upload) as mock_push:
        mock_inference_pipeline.push_to_hub("fake/repo-id", private=True)

    mock_push.assert_called_once()
    assert captured["repo_id"] == "fake/repo-id"
    assert captured["private"] is True
    assert "head.safetensors" in captured["files"]  # type: ignore[operator]
