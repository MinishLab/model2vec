from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

logger = logging.getLogger(__name__)

from model2vec.model import StaticModel
from tests.integration.pretrained_model_metrics import (
    PRETRAINED_MODELS,
    baseline_path_for,
    compute_metrics,
    load_static_model,
)

_ATTRIBUTE_FIELDS = (
    "full_vocab_size",
    "embedding_rows",
    "embedding_dim",
    "embedding_dtype",
    "token_order_hash",
    "first_tokens",
    "last_tokens",
    "median_token_length",
    "unk_token_id",
    "normalize",
    "base_model_name",
    "language",
    "vocabulary_quantization",
    "has_weights",
    "has_token_mapping",
    "tokenizer_type",
    "config",
)


@pytest.fixture(scope="module", params=sorted(PRETRAINED_MODELS), ids=sorted(PRETRAINED_MODELS))
def model_name(request: pytest.FixtureRequest) -> str:
    """A published model id, parametrizing the whole module."""
    return request.param


@pytest.fixture(scope="module")
def baseline(model_name: str) -> dict[str, Any]:
    """Load the stored golden baseline for this model."""
    path = baseline_path_for(model_name)
    if not path.exists():
        pytest.fail(f"No baseline found at {path}. Generate one with `make test-integration-pretrained-update`.")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def model(model_name: str) -> StaticModel:
    """Download the published model once for the whole module."""
    return load_static_model(model_name)


@pytest.fixture(scope="module")
def current_metrics(model: StaticModel) -> dict[str, Any]:
    """Compute the same metrics as the baseline for the freshly loaded model."""
    return compute_metrics(model)


def test_all_attributes_are_loaded(model: StaticModel) -> None:
    """Every attribute a `StaticModel` is expected to expose must be present and well-formed after loading."""
    assert isinstance(model.embedding, np.ndarray)
    assert model.embedding.ndim == 2
    assert len(model.tokens) == model.embedding.shape[0]
    vocab = model.tokenizer.get_vocab()
    assert model.tokens == tuple(sorted(vocab, key=lambda token: vocab[token]))
    assert model.dim == model.embedding.shape[1]
    assert model.embedding_dtype == np.dtype(model.embedding.dtype).name
    assert isinstance(model.config, dict) and model.config
    assert isinstance(model.normalize, bool)
    assert isinstance(model.median_token_length, int) and model.median_token_length > 0
    assert model.unk_token_id is None or isinstance(model.unk_token_id, int)
    assert model.base_model_name is None or isinstance(model.base_model_name, str)
    assert model.language is None or isinstance(model.language, list)


def test_attributes_match_baseline(model_name: str, baseline: dict[str, Any], current_metrics: dict[str, Any]) -> None:
    """Vocab size, token order/identity, dtype, and every loaded attribute must match the baseline exactly."""
    expected = baseline["metrics"]
    for field in _ATTRIBUTE_FIELDS:
        assert current_metrics[field] == expected[field], (
            f"[{model_name}] '{field}' drifted from the baseline: "
            f"expected {expected[field]!r}, got {current_metrics[field]!r}. "
            "If this is intentional, run `make test-integration-pretrained-update` and review the JSON diff."
        )


def test_vocab_size_and_order_match_baseline(
    model_name: str, baseline: dict[str, Any], current_metrics: dict[str, Any]
) -> None:
    """The vocabulary must be the same size and in the same order as the baseline."""
    expected = baseline["metrics"]
    assert current_metrics["full_vocab_size"] == expected["full_vocab_size"], model_name
    assert current_metrics["embedding_rows"] == expected["embedding_rows"], model_name
    assert current_metrics["token_order_hash"] == expected["token_order_hash"], model_name
    assert current_metrics["first_tokens"] == expected["first_tokens"], model_name
    assert current_metrics["last_tokens"] == expected["last_tokens"], model_name


def test_embedding_distribution_matches_baseline(
    model_name: str, baseline: dict[str, Any], current_metrics: dict[str, Any]
) -> None:
    """The embedding matrix's rank and mean/std shouldn't drift from the baseline."""
    expected = baseline["metrics"]
    assert abs(current_metrics["embedding_rank"] - expected["embedding_rank"]) <= 1, model_name
    assert current_metrics["embedding_mean"] == pytest.approx(expected["embedding_mean"], abs=1e-4), model_name
    assert current_metrics["embedding_std"] == pytest.approx(expected["embedding_std"], rel=0.02, abs=1e-4), model_name
    assert current_metrics["embedding_row_norm_mean"] == pytest.approx(
        expected["embedding_row_norm_mean"], rel=0.02, abs=1e-4
    ), model_name


def test_mteb_sts_scores_match_baseline(
    model_name: str, baseline: dict[str, Any], current_metrics: dict[str, Any]
) -> None:
    """MTEB STS scores for a published model must stay within a small tolerance of the golden baseline."""
    expected_scores = baseline["metrics"]["mteb_sts_scores"]
    actual_scores = current_metrics["mteb_sts_scores"]
    for task_name, expected_score in expected_scores.items():
        actual_score = actual_scores[task_name]
        assert actual_score == pytest.approx(expected_score, abs=0.01), (
            f"[{model_name}] MTEB '{task_name}' STS score drifted from baseline: "
            f"{expected_score:.4f} -> {actual_score:.4f}"
        )


def test_encoding_speed_is_measurable(
    model_name: str, baseline: dict[str, Any], current_metrics: dict[str, Any]
) -> None:
    """Record encoding throughput for reference. Never fails on speed: it is machine-dependent and a target to optimize."""
    expected = baseline["metrics"]["encoding_speed"]
    actual = current_metrics["encoding_speed"]
    logger.info(
        f"[{model_name}] encoding speed: "
        f"{actual['sentences_per_second']:.0f} sent/s ({expected['sentences_per_second']:.0f} baseline), "
        f"{actual['tokens_per_second']:.0f} tok/s ({expected['tokens_per_second']:.0f} baseline)"
    )
    assert actual["sentences_per_second"] > 0
    assert actual["tokens_per_second"] > 0


def test_save_and_load_roundtrip(model: StaticModel, tmp_path: Path) -> None:
    """Saving and reloading a published model must not change its tokens or embeddings."""
    save_path = tmp_path / "pretrained_model"
    model.save_pretrained(save_path)
    loaded_model = StaticModel.from_pretrained(save_path)

    assert loaded_model.tokens == model.tokens
    assert loaded_model.unk_token_id == model.unk_token_id
    assert loaded_model.normalize == model.normalize
    np.testing.assert_array_equal(loaded_model.embedding, model.embedding)
