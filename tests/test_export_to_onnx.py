from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation

# The exporter lives in scripts/, which is not an installed package.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from export_to_onnx import TorchStaticModelPipeline  # noqa: E402


def _export(torch_model: TorchStaticModelPipeline, texts: list[str], path: Path) -> np.ndarray:
    """Export the wrapped pipeline to ONNX and run inference on the given texts."""
    torch_model.eval()
    input_ids, offsets = torch_model.tokenize(texts)
    torch.onnx.export(
        torch_model,
        (input_ids, offsets),
        str(path),
        opset_version=14,
        input_names=["input_ids", "offsets"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "num_tokens"}, "offsets": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    session = ort.InferenceSession(str(path))
    return session.run(None, {"input_ids": input_ids.numpy(), "offsets": offsets.numpy()})[0]


def test_pipeline_onnx_matches_predict_proba(mock_inference_pipeline: StaticModelPipeline, tmp_path: Path) -> None:
    """The exported classifier ONNX graph should reproduce the pipeline's probabilities."""
    texts = ["dog", "cat", "dog cat"]
    torch_model = TorchStaticModelPipeline(mock_inference_pipeline)

    onnx_output = _export(torch_model, texts, tmp_path / "model.onnx")
    expected = mock_inference_pipeline.predict_proba(texts, use_multiprocessing=False)

    assert onnx_output.shape == expected.shape
    np.testing.assert_allclose(onnx_output, expected, atol=1e-4)


def test_pipeline_onnx_matches_projector(
    mock_inference_pipeline_projector: StaticModelPipeline, tmp_path: Path
) -> None:
    """An identity-head (regressor/projector) exports raw predictions, not probabilities."""
    assert mock_inference_pipeline_projector.head.activation == Activation.IDENTITY
    texts = ["dog", "cat"]
    torch_model = TorchStaticModelPipeline(mock_inference_pipeline_projector)

    onnx_output = _export(torch_model, texts, tmp_path / "model.onnx")
    expected = mock_inference_pipeline_projector.predict(texts, use_multiprocessing=False)

    assert onnx_output.shape == expected.shape
    np.testing.assert_allclose(onnx_output, expected, atol=1e-4)
