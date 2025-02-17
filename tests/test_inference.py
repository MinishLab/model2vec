import os
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from model2vec.inference import StaticModelPipeline


def test_init_predict(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test successful init and predict with StaticModelPipeline."""
    target: list[str] | list[list[str]]
    if mock_inference_pipeline.multilabel:
        if isinstance(mock_inference_pipeline.classes_[0], str):
            target = [["a", "b"]]
        else:
            target = [[0, 1]]  # type: ignore
    else:
        if isinstance(mock_inference_pipeline.classes_[0], str):
            target = ["b"]
        else:
            target = [1]  # type: ignore
    assert mock_inference_pipeline.predict("dog").tolist() == target
    assert mock_inference_pipeline.predict(["dog"]).tolist() == target


def test_init_predict_proba(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test successful init and predict_proba with StaticModelPipeline."""
    assert mock_inference_pipeline.predict_proba("dog").argmax() == 1
    assert mock_inference_pipeline.predict_proba(["dog"]).argmax(1).tolist() == [1]


def test_init_evaluate(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test successful init and evaluate with StaticModelPipeline."""
    target: list[str] | list[list[str]]
    if mock_inference_pipeline.multilabel:
        if isinstance(mock_inference_pipeline.classes_[0], str):
            target = [["a", "b"]]
        else:
            target = [[0, 1]]  # type: ignore
    else:
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
        if mock_inference_pipeline.multilabel:
            if isinstance(mock_inference_pipeline.classes_[0], str):
                target = [["a", "b"]]
            else:
                target = [[0, 1]]  # type: ignore
        else:
            if isinstance(mock_inference_pipeline.classes_[0], str):
                target = ["b"]
            else:
                target = [1]  # type: ignore
        assert loaded.predict("dog").tolist() == target
        assert loaded.predict(["dog"]).tolist() == target
        assert loaded.predict_proba("dog").argmax() == 1
        assert loaded.predict_proba(["dog"]).argmax(1).tolist() == [1]


@patch("model2vec.inference.model._DEFAULT_TRUST_PATTERN", re.compile("torch"))
def test_roundtrip_save_mock_trust_pattern(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test saving and loading the pipeline."""
    with TemporaryDirectory() as temp_dir:
        mock_inference_pipeline.save_pretrained(temp_dir)
        with pytest.raises(ValueError):
            StaticModelPipeline.from_pretrained(temp_dir)


def test_roundtrip_save_file_gone(mock_inference_pipeline: StaticModelPipeline) -> None:
    """Test saving and loading the pipeline."""
    with TemporaryDirectory() as temp_dir:
        mock_inference_pipeline.save_pretrained(temp_dir)
        # Rename the file to abc.pipeline, so that it looks like it was downloaded from the hub
        os.unlink(os.path.join(temp_dir, "pipeline.skops"))
        with pytest.raises(FileNotFoundError):
            StaticModelPipeline.from_pretrained(temp_dir)
