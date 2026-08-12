import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
import skops.io
from huggingface_hub.errors import EntryNotFoundError
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model2vec.inference.mlp import Activation, Layer, MLPHead
from model2vec.inference.model import StaticModelPipeline, convert_legacy_pipeline
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


def _dump_legacy_pipeline(temp_dir: str, model: StaticModel, legacy_pipeline: object) -> None:
    model.save_pretrained(temp_dir)
    skops.io.dump(legacy_pipeline, os.path.join(temp_dir, "pipeline.skops"))


def test_convert_legacy_pipeline_softmax(mock_static_model: StaticModel) -> None:
    """Test converting a legacy single-label classifier pipeline."""
    rng = np.random.RandomState(0)
    X = rng.randn(30, mock_static_model.dim)
    y = np.array(["a", "b", "c"] * 10)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1, random_state=0).fit(X, y)
    legacy_pipeline = make_pipeline(mlp)

    with TemporaryDirectory() as temp_dir:
        _dump_legacy_pipeline(temp_dir, mock_static_model, legacy_pipeline)
        converted = convert_legacy_pipeline(temp_dir)

    assert converted.head.activation == Activation.SOFTMAX
    encoded = mock_static_model.encode(["dog", "cat"])
    assert converted.predict(["dog", "cat"]).tolist() == legacy_pipeline.predict(encoded).tolist()


def test_convert_legacy_pipeline_sigmoid(mock_static_model: StaticModel) -> None:
    """Test converting a legacy multilabel classifier pipeline."""
    rng = np.random.RandomState(0)
    X = rng.randn(30, mock_static_model.dim)
    y = rng.randint(0, 2, size=(30, 3))
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1, random_state=0).fit(X, y)
    legacy_pipeline = make_pipeline(mlp)

    with TemporaryDirectory() as temp_dir:
        _dump_legacy_pipeline(temp_dir, mock_static_model, legacy_pipeline)
        converted = convert_legacy_pipeline(temp_dir)

    assert converted.head.activation == Activation.SIGMOID
    encoded = mock_static_model.encode(["dog", "cat"])
    assert np.allclose(converted.predict_proba(["dog", "cat"]), legacy_pipeline.predict_proba(encoded))


def test_convert_legacy_pipeline_identity(mock_static_model: StaticModel) -> None:
    """Test converting a legacy regressor (projector) pipeline."""
    rng = np.random.RandomState(0)
    X = rng.randn(30, mock_static_model.dim)
    y = rng.randn(30, 5)
    mlp = MLPRegressor(hidden_layer_sizes=(8,), max_iter=1, random_state=0).fit(X, y)
    legacy_pipeline = make_pipeline(mlp)

    with TemporaryDirectory() as temp_dir:
        _dump_legacy_pipeline(temp_dir, mock_static_model, legacy_pipeline)
        converted = convert_legacy_pipeline(temp_dir)

    assert converted.head.activation == Activation.IDENTITY
    assert converted.classes_ is None
    encoded = mock_static_model.encode(["dog", "cat"])
    assert np.allclose(converted.predict(["dog", "cat"]), legacy_pipeline.predict(encoded))


def test_convert_legacy_pipeline_file_gone(mock_static_model: StaticModel) -> None:
    """Test that a missing pipeline.skops file raises a clear error."""
    with TemporaryDirectory() as temp_dir:
        mock_static_model.save_pretrained(temp_dir)
        with pytest.raises(FileNotFoundError):
            convert_legacy_pipeline(temp_dir)


def test_convert_legacy_pipeline_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a helpful error is raised when scikit-learn/skops aren't installed."""
    monkeypatch.setitem(sys.modules, "skops", None)
    monkeypatch.setitem(sys.modules, "skops.io", None)

    with pytest.raises(ImportError, match="scikit-learn"):
        convert_legacy_pipeline("some/path")


def test_from_pretrained_legacy_fallback_local(mock_static_model: StaticModel) -> None:
    """Test that from_pretrained falls back to a local legacy pipeline.skops file, with a warning."""
    rng = np.random.RandomState(0)
    X = rng.randn(30, mock_static_model.dim)
    y = np.array(["a", "b", "c"] * 10)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1, random_state=0).fit(X, y)
    legacy_pipeline = make_pipeline(mlp)

    with TemporaryDirectory() as temp_dir:
        _dump_legacy_pipeline(temp_dir, mock_static_model, legacy_pipeline)
        with pytest.warns(UserWarning, match="legacy"):
            loaded = StaticModelPipeline.from_pretrained(temp_dir)

    assert loaded.head.activation == Activation.SOFTMAX
    encoded = mock_static_model.encode(["dog", "cat"])
    assert loaded.predict(["dog", "cat"]).tolist() == legacy_pipeline.predict(encoded).tolist()


def test_from_pretrained_legacy_fallback_hub(mock_static_model: StaticModel) -> None:
    """Test that from_pretrained falls back to a hub-downloaded legacy pipeline, with a warning."""
    rng = np.random.RandomState(0)
    X = rng.randn(30, mock_static_model.dim)
    y = rng.randn(30, 4)
    mlp = MLPRegressor(hidden_layer_sizes=(8,), max_iter=1, random_state=0).fit(X, y)
    legacy_pipeline = make_pipeline(mlp)

    with TemporaryDirectory() as temp_dir:
        mock_static_model.save_pretrained(temp_dir)
        pipeline_skops_path = os.path.join(temp_dir, "pipeline.skops")
        skops.io.dump(legacy_pipeline, pipeline_skops_path)
        downloaded_model = StaticModel.from_pretrained(temp_dir)

        def _fake_download(repo_id: str, filename: str, token: str | None = None) -> str:
            if filename == "head.safetensors":
                raise EntryNotFoundError("no head.safetensors")
            return pipeline_skops_path

        with (
            patch("model2vec.inference.model.huggingface_hub.hf_hub_download", side_effect=_fake_download),
            patch("model2vec.inference.model.StaticModel.from_pretrained", return_value=downloaded_model),
            pytest.warns(UserWarning, match="legacy"),
        ):
            loaded = StaticModelPipeline.from_pretrained("fake/repo-id")

    assert loaded.head.activation == Activation.IDENTITY
    encoded = mock_static_model.encode(["dog", "cat"])
    assert np.allclose(loaded.predict(["dog", "cat"]), legacy_pipeline.predict(encoded))


def test_convert_legacy_pipeline_untrusted_type(mock_static_model: StaticModel) -> None:
    """Test that an untrusted type in the legacy pipeline is rejected by default."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, mock_static_model.dim)
    y = np.array(["a", "b"] * 5)
    legacy_pipeline = make_pipeline(MLPClassifier(hidden_layer_sizes=(4,), max_iter=1, random_state=0).fit(X, y))

    with TemporaryDirectory() as temp_dir:
        _dump_legacy_pipeline(temp_dir, mock_static_model, legacy_pipeline)
        with patch("skops.io.get_untrusted_types", return_value=["evil.Type"]):
            with pytest.raises(ValueError, match="Untrusted type"):
                convert_legacy_pipeline(temp_dir)


def test_convert_legacy_pipeline_unsupported_head(mock_static_model: StaticModel) -> None:
    """Test that a non-MLP legacy head raises a clear error."""
    legacy_pipeline = make_pipeline(StandardScaler())

    with TemporaryDirectory() as temp_dir:
        _dump_legacy_pipeline(temp_dir, mock_static_model, legacy_pipeline)
        with pytest.raises(ValueError, match="Unsupported legacy head type"):
            convert_legacy_pipeline(temp_dir)


def test_convert_legacy_pipeline_unsupported_activation(mock_static_model: StaticModel) -> None:
    """Test that a non-relu hidden activation raises a clear error."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, mock_static_model.dim)
    y = np.array(["a", "b"] * 5)
    mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", max_iter=1, random_state=0).fit(X, y)
    legacy_pipeline = make_pipeline(mlp)

    with TemporaryDirectory() as temp_dir:
        _dump_legacy_pipeline(temp_dir, mock_static_model, legacy_pipeline)
        with pytest.raises(ValueError, match="Unsupported hidden activation"):
            convert_legacy_pipeline(temp_dir)


def test_save_pipeline_base_model_name_variants(mock_static_model: StaticModel) -> None:
    """Test that _save_pipeline resolves both list- and str-valued base_model_name."""
    head = MLPHead(layers=[Layer(weight=np.eye(3), bias=np.zeros(3))], activation=Activation.IDENTITY)

    for base_model_name, expected in [(["a/b", "c/d"], "a/b"), ("a/b", "a/b")]:
        model = StaticModel(
            vectors=mock_static_model.embedding,
            tokenizer=mock_static_model.tokenizer,
            base_model_name=base_model_name,  # type: ignore[arg-type]
        )
        pipeline = StaticModelPipeline(model, head)
        with TemporaryDirectory() as temp_dir:
            pipeline.save_pretrained(temp_dir)
            readme = (Path(temp_dir) / "README.md").read_text()
        assert expected in readme
