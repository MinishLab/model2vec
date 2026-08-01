"""Regression test for the distillation pipeline.

Downloads real sentence-transformers (see `BASE_MODELS`), distills several StaticModel variants
from each, and compares the result against a stored JSON baseline
(`tests/integration/data/<model>_baseline.json`): vocabulary size, token order, embedding matrix
rank/distribution, and semantic similarity scores. This catches regressions that unit tests (which
run against a mocked transformer) can't, e.g. a tokenizer/embedding misalignment, a degenerate
(rank-collapsed) embedding matrix, or a silent drop in semantic quality -- and, by covering more
than one base model, regressions specific to a particular tokenizer family (e.g. BPE vs. wordpiece).

This suite requires network access and real forward passes over a full vocabulary, so it is
intentionally excluded from `make test` and must be run explicitly with `make test-integration`.

If a change intentionally alters distillation output, regenerate the baselines with
`make test-integration-update` and review the JSON diff before committing it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from model2vec.distill import distill_from_model
from model2vec.model import StaticModel
from tests.integration._distill_metrics import (
    BASE_MODELS,
    CONFIGS,
    baseline_path_for,
    compute_metrics,
    distill_all,
    load_base_model_and_tokenizer,
)

_EXACT_FIELDS = (
    "full_vocab_size",
    "embedding_rows",
    "embedding_dim",
    "token_order_hash",
    "first_tokens",
    "last_tokens",
)


@pytest.fixture(scope="module", params=sorted(BASE_MODELS), ids=sorted(BASE_MODELS))
def model_name(request: pytest.FixtureRequest) -> str:
    """A short base-model name (a key of `BASE_MODELS`), parametrizing the whole module."""
    return request.param


@pytest.fixture(scope="module")
def baseline(model_name: str) -> dict[str, Any]:
    """Load the stored golden baseline for this base model."""
    path = baseline_path_for(model_name)
    if not path.exists():
        pytest.fail(f"No baseline found at {path}. Generate one with `make test-integration-update`.")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def base_model_and_tokenizer(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Download this module's base sentence-transformer once, for reuse across every test in it."""
    return load_base_model_and_tokenizer(BASE_MODELS[model_name])


@pytest.fixture(scope="module")
def distilled_models(
    base_model_and_tokenizer: tuple[PreTrainedModel, PreTrainedTokenizerFast],
) -> dict[str, StaticModel]:
    """Distill every configured variant from the base model, once for the whole module."""
    model, tokenizer = base_model_and_tokenizer
    return distill_all(model, tokenizer)


@pytest.fixture(scope="module")
def current_metrics(distilled_models: dict[str, StaticModel]) -> dict[str, dict[str, Any]]:
    """Compute the same metrics as the baseline for the freshly distilled models."""
    return {name: compute_metrics(static_model) for name, static_model in distilled_models.items()}


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_structural_metrics_match_baseline(
    config_name: str, model_name: str, baseline: dict[str, Any], current_metrics: dict[str, dict[str, Any]]
) -> None:
    """Vocab size, embedding shape, token order, and token identity must match the baseline exactly."""
    expected = baseline["configs"][config_name]
    actual = current_metrics[config_name]
    for field in _EXACT_FIELDS:
        assert actual[field] == expected[field], (
            f"[{model_name}/{config_name}] '{field}' drifted from the baseline: "
            f"expected {expected[field]!r}, got {actual[field]!r}. "
            "If this is intentional, run `make test-integration-update` and review the JSON diff."
        )


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_embedding_rank_matches_baseline(
    config_name: str, model_name: str, baseline: dict[str, Any], current_metrics: dict[str, dict[str, Any]]
) -> None:
    """A rank drop vs. the baseline means the pooling/PCA/quantization step degenerated."""
    expected_rank = baseline["configs"][config_name]["embedding_rank"]
    actual_rank = current_metrics[config_name]["embedding_rank"]
    assert abs(actual_rank - expected_rank) <= 1, (
        f"[{model_name}/{config_name}] embedding rank {actual_rank} vs baseline {expected_rank}"
    )


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_embedding_distribution_matches_baseline(
    config_name: str, model_name: str, baseline: dict[str, Any], current_metrics: dict[str, dict[str, Any]]
) -> None:
    """The embedding matrix's mean/std shouldn't drift far from the baseline (small tolerance for numerical noise)."""
    expected = baseline["configs"][config_name]
    actual = current_metrics[config_name]
    assert actual["embedding_mean"] == pytest.approx(expected["embedding_mean"], abs=1e-3), model_name
    assert actual["embedding_std"] == pytest.approx(expected["embedding_std"], rel=0.05, abs=1e-3), model_name


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_semantic_similarity_matches_baseline(
    config_name: str, model_name: str, baseline: dict[str, Any], current_metrics: dict[str, dict[str, Any]]
) -> None:
    """Guard against silently destroyed performance."""
    expected_scores = baseline["configs"][config_name]["similarity_scores"]
    actual_scores = current_metrics[config_name]["similarity_scores"]
    for pair_name, expected_score in expected_scores.items():
        actual_score = actual_scores[pair_name]
        assert actual_score == pytest.approx(expected_score, abs=0.05), (
            f"[{model_name}/{config_name}] similarity for '{pair_name}' drifted from baseline: "
            f"{expected_score:.4f} -> {actual_score:.4f}"
        )


def test_distillation_is_deterministic(
    base_model_and_tokenizer: tuple[PreTrainedModel, PreTrainedTokenizerFast],
) -> None:
    """Distilling twice with identical parameters must yield identical embeddings and vocabularies."""
    model, tokenizer = base_model_and_tokenizer
    kwargs = CONFIGS["custom_vocab"]

    first = distill_from_model(model=copy.deepcopy(model), tokenizer=tokenizer, **kwargs)
    second = distill_from_model(model=copy.deepcopy(model), tokenizer=tokenizer, **kwargs)

    assert first.tokens == second.tokens
    np.testing.assert_allclose(first.embedding, second.embedding, rtol=1e-5, atol=1e-6)


def test_save_and_load_roundtrip(distilled_models: dict[str, StaticModel], tmp_path: Path) -> None:
    """Saving and reloading a distilled model must not change its tokens or embeddings."""
    model = distilled_models["subword"]
    save_path = tmp_path / "distilled_model"
    model.save_pretrained(save_path)
    loaded_model = StaticModel.from_pretrained(save_path)

    assert loaded_model.tokens == model.tokens
    np.testing.assert_array_equal(loaded_model.embedding, model.embedding)
