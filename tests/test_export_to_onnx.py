from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from model2vec import StaticModel
from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation
from model2vec.onnx import (
    TorchStaticModelPipeline,
    _dynamic_shapes,
    _export_onnx,
    _resolve_pad_token_id,
    export_model_to_onnx,
)


def _tokenize(pipeline: StaticModelPipeline, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad-tokenize texts into (input_ids, attention_mask) tensors, transformers-style."""
    tokenizer = pipeline.model.tokenizer
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    tokenizer.no_padding()
    input_ids = torch.tensor([e.ids for e in encodings], dtype=torch.long)
    attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long)
    return input_ids, attention_mask


def _export(
    torch_model: TorchStaticModelPipeline, input_ids: torch.Tensor, attention_mask: torch.Tensor, path: Path
) -> np.ndarray:
    """Export the wrapped pipeline to ONNX and run inference on the given inputs."""
    torch_model.eval()
    _export_onnx(
        torch_model,
        (input_ids, attention_mask),
        str(path),
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_shapes=_dynamic_shapes(),
    )
    session = ort.InferenceSession(str(path))
    output = session.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})[0]
    assert isinstance(output, np.ndarray)
    return output


def test_pipeline_onnx_matches_predict_proba(mock_inference_pipeline: StaticModelPipeline, tmp_path: Path) -> None:
    """The exported classifier ONNX graph should reproduce the pipeline's probabilities."""
    texts = ["dog", "cat", "dog cat"]
    torch_model = TorchStaticModelPipeline(mock_inference_pipeline)
    input_ids, attention_mask = _tokenize(mock_inference_pipeline, texts)

    onnx_output = _export(torch_model, input_ids, attention_mask, tmp_path / "model.onnx")
    expected = mock_inference_pipeline.predict_proba(texts, use_multiprocessing=False)

    assert onnx_output.shape == expected.shape
    np.testing.assert_allclose(onnx_output, expected, atol=1e-4)


def test_export_model_to_onnx_encoder(mock_static_model: StaticModel, tmp_path: Path) -> None:
    """A plain StaticModel is exported with a config.json and special_tokens_map.json alongside it."""
    save_path = tmp_path / "export"
    export_model_to_onnx(mock_static_model, save_path)

    assert (save_path / "model.onnx").exists()
    tokenizer_model = TokenizerModel.from_tokenizer(mock_static_model.tokenizer)
    expected_pad_token_id = _resolve_pad_token_id(mock_static_model.tokenizer, tokenizer_model)

    config = json.loads((save_path / "config.json").read_text())
    assert config["pad_token_id"] == expected_pad_token_id

    special_tokens_map = json.loads((save_path / "special_tokens_map.json").read_text())
    assert special_tokens_map["pad_token"] == mock_static_model.tokenizer.id_to_token(expected_pad_token_id)


def test_export_model_to_onnx_pipeline(mock_inference_pipeline: StaticModelPipeline, tmp_path: Path) -> None:
    """A StaticModelPipeline is exported with a config.json and special_tokens_map.json alongside it."""
    save_path = tmp_path / "export"
    export_model_to_onnx(mock_inference_pipeline, save_path)

    assert (save_path / "model.onnx").exists()
    tokenizer_model = TokenizerModel.from_tokenizer(mock_inference_pipeline.model.tokenizer)
    expected_pad_token_id = _resolve_pad_token_id(mock_inference_pipeline.model.tokenizer, tokenizer_model)

    config = json.loads((save_path / "config.json").read_text())
    assert config["pad_token_id"] == expected_pad_token_id

    special_tokens_map = json.loads((save_path / "special_tokens_map.json").read_text())
    assert special_tokens_map["pad_token"] == mock_inference_pipeline.model.tokenizer.id_to_token(expected_pad_token_id)


def test_resolve_pad_token_id_falls_back_to_unk_when_no_pad_registered() -> None:
    """When a tokenizer has no registered pad token and no literal "[PAD]" entry, fall back to unk, not position 0."""
    vocab = ["!", "hello", "world", "[UNK]"]
    tokenizer = Tokenizer(BPE(vocab={t: i for i, t in enumerate(vocab)}, merges=[], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
    tokenizer_model = TokenizerModel.from_tokenizer(tokenizer)
    assert tokenizer_model.pad_token_id is None
    assert tokenizer.token_to_id("[PAD]") is None

    assert _resolve_pad_token_id(tokenizer, tokenizer_model) == tokenizer.token_to_id("[UNK]")


def test_pipeline_onnx_matches_projector(
    mock_inference_pipeline_projector: StaticModelPipeline, tmp_path: Path
) -> None:
    """An identity-head (regressor/projector) exports raw predictions, not probabilities."""
    assert mock_inference_pipeline_projector.head.activation == Activation.IDENTITY
    texts = ["dog", "cat"]
    torch_model = TorchStaticModelPipeline(mock_inference_pipeline_projector)
    input_ids, attention_mask = _tokenize(mock_inference_pipeline_projector, texts)

    onnx_output = _export(torch_model, input_ids, attention_mask, tmp_path / "model.onnx")
    expected = mock_inference_pipeline_projector.predict(texts, use_multiprocessing=False)

    assert onnx_output.shape == expected.shape
    np.testing.assert_allclose(onnx_output, expected, atol=1e-4)
