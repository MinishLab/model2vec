from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

from model2vec import StaticModel
from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation
from model2vec.model import DEFAULT_MAX_LENGTH
from model2vec.onnx import (
    TorchStaticModel,
    TorchStaticModelPipeline,
    _dynamic_shapes,
    _export_onnx,
    _resolve_pad_token_id,
    _save_tokenizer_and_config,
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


def test_resolve_pad_token_id_falls_back_to_zero_when_no_pad_or_unk() -> None:
    """With no registered pad token, no literal "[PAD]" entry, and no unk token, fall back to id 0."""
    vocab = ["!", "hello", "world"]
    tokenizer = Tokenizer(BPE(vocab={t: i for i, t in enumerate(vocab)}, merges=[]))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
    tokenizer_model = TokenizerModel.from_tokenizer(tokenizer)
    assert tokenizer_model.pad_token_id is None
    assert tokenizer.token_to_id("[PAD]") is None
    assert tokenizer_model.unk_token_id is None

    assert _resolve_pad_token_id(tokenizer, tokenizer_model) == 0


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


def test_save_tokenizer_and_config_removes_post_processor_by_default(
    mock_static_model: StaticModel, tmp_path: Path
) -> None:
    """`remove_post_processor=True` (the default) strips special-token insertion from the saved tokenizer."""
    tokenizer_model = TokenizerModel.from_tokenizer(mock_static_model.tokenizer)
    assert tokenizer_model.post_processor is not None

    _save_tokenizer_and_config(
        mock_static_model.tokenizer, tmp_path, remove_post_processor=True, max_length=mock_static_model.max_length
    )

    saved_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    with_special = saved_tokenizer("hello", add_special_tokens=True)["input_ids"]
    without_special = saved_tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert with_special == without_special


def test_save_tokenizer_and_config_keeps_post_processor_when_disabled(
    mock_static_model: StaticModel, tmp_path: Path
) -> None:
    """`remove_post_processor=False` preserves special-token insertion in the saved tokenizer."""
    tokenizer_model = TokenizerModel.from_tokenizer(mock_static_model.tokenizer)
    assert tokenizer_model.post_processor is not None

    _save_tokenizer_and_config(
        mock_static_model.tokenizer, tmp_path, remove_post_processor=False, max_length=mock_static_model.max_length
    )

    saved_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    with_special = saved_tokenizer("hello", add_special_tokens=True)["input_ids"]
    without_special = saved_tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert with_special != without_special


def test_save_tokenizer_and_config_warns_when_removing_post_processor(
    mock_static_model: StaticModel, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A warning is logged when a post processor is actually present and removed."""
    with caplog.at_level(logging.WARNING, logger="model2vec.onnx"):
        _save_tokenizer_and_config(
            mock_static_model.tokenizer, tmp_path, remove_post_processor=True, max_length=mock_static_model.max_length
        )

    assert "removing a post processor" in caplog.text


def test_save_tokenizer_and_config_no_warning_without_post_processor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No warning is logged when the tokenizer has no post processor to begin with."""
    vocab = ["!", "hello", "world", "[UNK]"]
    tokenizer = Tokenizer(BPE(vocab={t: i for i, t in enumerate(vocab)}, merges=[], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
    assert TokenizerModel.from_tokenizer(tokenizer).post_processor is None

    with caplog.at_level(logging.WARNING, logger="model2vec.onnx"):
        _save_tokenizer_and_config(tokenizer, tmp_path, remove_post_processor=True, max_length=512)

    assert "removing a post processor" not in caplog.text


def test_save_tokenizer_and_config_defaults_max_length_when_none(
    mock_static_model: StaticModel, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A `None` max_length is warned about and replaced with `DEFAULT_MAX_LENGTH` in the saved tokenizer."""
    with caplog.at_level(logging.WARNING, logger="model2vec.onnx"):
        _save_tokenizer_and_config(mock_static_model.tokenizer, tmp_path, remove_post_processor=True, max_length=None)

    assert "no max length" in caplog.text

    saved_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert saved_tokenizer.model_max_length == DEFAULT_MAX_LENGTH


def test_export_model_to_onnx_remove_post_processor_default_true(
    mock_static_model: StaticModel, tmp_path: Path
) -> None:
    """`export_model_to_onnx` defaults to removing the post processor, matching the documented default."""
    export_model_to_onnx(mock_static_model, tmp_path)

    saved_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    with_special = saved_tokenizer("hello", add_special_tokens=True)["input_ids"]
    without_special = saved_tokenizer("hello", add_special_tokens=False)["input_ids"]
    assert with_special == without_special


def _encoder_onnx_output(model: StaticModel, texts: list[str], path: Path) -> np.ndarray:
    """Export a plain encoder to ONNX and run it on pad-tokenized `texts`."""
    tokenizer = model.tokenizer
    tokenizer.enable_padding(pad_id=0, pad_token=tokenizer.id_to_token(0))
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    tokenizer.no_padding()
    input_ids = torch.tensor([e.ids for e in encodings], dtype=torch.long)
    attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long)

    torch_model = TorchStaticModel(model)
    torch_model.eval()
    _export_onnx(
        torch_model,
        (input_ids, attention_mask),
        str(path),
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_shapes=_dynamic_shapes(),
    )
    session = ort.InferenceSession(str(path))
    output = session.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})[0]
    assert isinstance(output, np.ndarray)
    return output


def test_encoder_onnx_zeroes_unk_tokens(tmp_path: Path) -> None:
    """The exported encoder drops `[UNK]` token contributions, matching `StaticModel.encode`."""
    vocab = ["[PAD]", "dog", "cat", "fish", "[UNK]"]
    tokenizer = Tokenizer(
        BPE(vocab={t: i for i, t in enumerate(vocab)}, merges=[], unk_token="[UNK]", ignore_merges=True)
    )
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
    vectors = np.random.RandomState(0).randn(len(vocab), 8).astype(np.float32)
    model = StaticModel(vectors=vectors, tokenizer=tokenizer, normalize=False)
    assert model.unk_token_id == vocab.index("[UNK]")

    texts = ["dog cat", "dog zzz cat", "fish zzz"]
    onnx_output = _encoder_onnx_output(model, texts, tmp_path / "model.onnx")
    expected = model.encode(texts)

    np.testing.assert_allclose(onnx_output, expected, atol=1e-5)


def test_encoder_onnx_without_unk_token(tmp_path: Path) -> None:
    """A tokenizer with no unk token (`unk_token_id is None`) exports and runs without masking."""
    vocab = ["[PAD]", "dog", "cat", "fish"]
    tokenizer = Tokenizer(BPE(vocab={t: i for i, t in enumerate(vocab)}, merges=[], ignore_merges=True))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
    vectors = np.random.RandomState(0).randn(len(vocab), 8).astype(np.float32)
    model = StaticModel(vectors=vectors, tokenizer=tokenizer, normalize=False)
    assert model.unk_token_id is None

    texts = ["dog cat", "fish dog", "cat"]
    onnx_output = _encoder_onnx_output(model, texts, tmp_path / "model.onnx")
    expected = model.encode(texts)

    np.testing.assert_allclose(onnx_output, expected, atol=1e-5)
