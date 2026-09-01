from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from skeletoken import TokenizerModel

from model2vec import StaticModel
from model2vec.onnx import _resolve_pad_token_id, export_model_to_onnx

_MODEL_NAME = "minishlab/potion-base-32m"

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Paris is the capital of France.",
    "I would like to order a large pizza with extra cheese.",
    "Machine learning models can be trained on large datasets.",
    "The weather today is sunny with a light breeze.",
    "She sells seashells by the seashore.",
    "This is a much longer sentence than the others, so padding and truncation are also exercised.",
    "hi",
]


@pytest.fixture(scope="module")
def model() -> StaticModel:
    """Download the real `potion-base-32m` model once for this module."""
    return StaticModel.from_pretrained(_MODEL_NAME)


@pytest.fixture(scope="module")
def export_dir(model: StaticModel, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the real model to ONNX once, for reuse across every test in this module."""
    save_path = tmp_path_factory.mktemp("potion_onnx_export")
    export_model_to_onnx(model, save_path)
    return save_path


@pytest.fixture(scope="module")
def onnx_session(export_dir: Path) -> ort.InferenceSession:
    """Load the exported ONNX graph into an onnxruntime session."""
    return ort.InferenceSession(str(export_dir / "model.onnx"))


def _tokenize(model: StaticModel, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Pad-tokenize texts into (input_ids, attention_mask) arrays, transformers-style."""
    tokenizer = model.tokenizer
    pad_token_id = _resolve_pad_token_id(tokenizer, TokenizerModel.from_tokenizer(tokenizer))
    tokenizer.enable_padding(pad_id=pad_token_id)
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    tokenizer.no_padding()
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    return input_ids, attention_mask


def test_onnx_export_matches_encode_on_real_sentences(model: StaticModel, onnx_session: ort.InferenceSession) -> None:
    """The ONNX graph exported from a real model should reproduce `StaticModel.encode` on real sentences."""
    input_ids, attention_mask = _tokenize(model, _SENTENCES)

    onnx_output = onnx_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    assert isinstance(onnx_output, np.ndarray)
    expected = model.encode(_SENTENCES, use_multiprocessing=False)

    assert onnx_output.shape == expected.shape
    np.testing.assert_allclose(onnx_output, expected, rtol=1e-4, atol=1e-4)


def test_onnx_export_writes_pad_token_id(model: StaticModel, export_dir: Path) -> None:
    """The exported config.json and special_tokens_map.json must carry the real model's pad token."""
    expected_pad_token_id = _resolve_pad_token_id(model.tokenizer, TokenizerModel.from_tokenizer(model.tokenizer))

    config = json.loads((export_dir / "config.json").read_text())
    assert config["pad_token_id"] == expected_pad_token_id

    special_tokens_map = json.loads((export_dir / "special_tokens_map.json").read_text())
    assert special_tokens_map["pad_token"] == model.tokenizer.id_to_token(expected_pad_token_id)
