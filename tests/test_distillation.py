# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from importlib import import_module
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import LogCaptureFixture
from transformers import BertTokenizerFast
from transformers.modeling_utils import PreTrainedModel

from model2vec.distill.distillation import distill, distill_from_model
from model2vec.distill.inference import PoolingType, create_embeddings, post_process_embeddings
from model2vec.model import StaticModel
from model2vec.tokenizer import clean_and_create_vocabulary

try:
    # For huggingface_hub>=0.25.0
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    # For huggingface_hub<0.25.0
    from huggingface_hub.utils._errors import RepositoryNotFoundError  # type: ignore

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "vocabulary, pca_dims, sif_coefficient",
    [
        (None, 256, 1e-4),  # Subword vocab, PCA applied, SIF on
        (["wordA", "wordB"], 4, None),  # Custom vocab, PCA applied, SIF off
        (None, "auto", None),  # Subword, PCA 'auto', SIF off
        (None, 1024, None),  # Subword, PCA set high, SIF off
        (None, None, 1e-4),  # No PCA, SIF on
        (None, 0.9, 1e-4),  # PCA as float (variance), SIF on
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill_from_model(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_berttokenizer: BertTokenizerFast,
    mock_transformer: PreTrainedModel,
    vocabulary: list[str] | None,
    pca_dims: int | None,
    sif_coefficient: float | None,
) -> None:
    """Test distill function with different parameters."""
    # Mock the return value of model_info to avoid calling the Hugging Face API
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})
    mock_auto_model.return_value = mock_transformer

    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=vocabulary,
        device="cpu",
        pca_dims=pca_dims,
        sif_coefficient=sif_coefficient,
        token_remove_pattern=None,
    )

    static_model2 = distill(
        model_name="tests/data/test_tokenizer",
        vocabulary=vocabulary,
        device="cpu",
        pca_dims=pca_dims,
        sif_coefficient=sif_coefficient,
        token_remove_pattern=None,
    )

    assert static_model.embedding.shape == static_model2.embedding.shape
    assert static_model.config == static_model2.config
    assert json.loads(static_model.tokenizer.to_str()) == json.loads(static_model2.tokenizer.to_str())
    assert static_model.base_model_name == static_model2.base_model_name


@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill_removal_pattern(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_berttokenizer: BertTokenizerFast,
    mock_transformer: PreTrainedModel,
) -> None:
    """Test the removal pattern."""
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})
    mock_auto_model.return_value = mock_transformer

    # The vocab size is 30522, but we remove 998 tokens: [CLS], [SEP], and [MASK], and all [unused] tokens.
    expected_vocab_size = mock_berttokenizer.vocab_size - 998

    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=None,
        device="cpu",
        token_remove_pattern=None,
    )
    assert len(static_model.embedding) == expected_vocab_size

    # No tokens removed, nonsensical pattern
    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=None,
        device="cpu",
        token_remove_pattern="£££££££££££££££££",
    )
    assert len(static_model.embedding) == expected_vocab_size

    # Weird pattern.
    with pytest.raises(ValueError):
        _ = distill_from_model(
            model=mock_transformer,
            tokenizer=mock_berttokenizer,
            vocabulary=None,
            device="cpu",
            token_remove_pattern="[...papapa",
        )


@pytest.mark.parametrize(
    "vocabulary, pca_dims, sif_coefficient, expected_shape",
    [
        (None, 256, None, (29524, 256)),  # PCA applied, SIF off
        (None, "auto", None, (29524, 768)),  # PCA 'auto', SIF off
        (None, "auto", 1e-4, (29524, 768)),  # PCA 'auto', SIF on
        (None, "auto", 0, None),  # invalid SIF (too low) -> raises
        (None, "auto", 1, None),  # invalid SIF (too high) -> raises
        (None, 1024, None, (29524, 768)),  # PCA set high (no reduction)
        (["wordA", "wordB"], 4, None, (29526, 4)),  # Custom vocab, PCA applied
        (None, None, None, (29524, 768)),  # No PCA, SIF off
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_transformer: PreTrainedModel,
    vocabulary: list[str] | None,
    pca_dims: int | None,
    sif_coefficient: float | None,
    expected_shape: tuple[int, int] | None,
) -> None:
    """Test distill function with different parameters."""
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})
    mock_auto_model.return_value = mock_transformer

    model_name = "tests/data/test_tokenizer"

    if sif_coefficient is not None and (sif_coefficient <= 0 or sif_coefficient >= 1):
        with pytest.raises(ValueError):
            _ = distill(
                model_name=model_name,
                vocabulary=vocabulary,
                device="cpu",
                pca_dims=pca_dims,
                sif_coefficient=sif_coefficient,
            )
    else:
        static_model = distill(
            model_name=model_name,
            vocabulary=vocabulary,
            device="cpu",
            pca_dims=pca_dims,
            sif_coefficient=sif_coefficient,
        )
        assert isinstance(static_model, StaticModel)
        assert static_model.embedding.shape == expected_shape
        assert "mock-model" in static_model.config["tokenizer_name"]
        assert static_model.tokenizer is not None


@patch.object(import_module("model2vec.distill.distillation"), "model_info")
def test_missing_modelinfo(
    mock_model_info: MagicMock,
    mock_transformer: PreTrainedModel,
    mock_berttokenizer: BertTokenizerFast,
) -> None:
    """Test that missing model info does not crash."""
    mock_model_info.side_effect = RepositoryNotFoundError("Model not found")
    static_model = distill_from_model(model=mock_transformer, tokenizer=mock_berttokenizer, device="cpu")
    assert static_model.language is None


@pytest.mark.parametrize(
    "embeddings, pca_dims, sif_coefficient, expected_shape",
    [
        (rng.random((1000, 768)), 256, None, (1000, 256)),  # PCA applied correctly
        (rng.random((1000, 768)), None, None, (1000, 768)),  # No PCA applied, dimensions unchanged
        (rng.random((1000, 768)), 256, 1e-4, (1000, 256)),  # PCA and SIF applied
        (rng.random((10, 768)), 256, 1e-4, (10, 768)),  # PCA dims > vocab size, no PCA applied
    ],
)
def test__post_process_embeddings(
    embeddings: np.ndarray, pca_dims: int | float | None, sif_coefficient: float | None, expected_shape: tuple[int, int]
) -> None:
    """Test the post_process_embeddings function."""
    original_embeddings = embeddings.copy()  # Copy embeddings to compare later

    # If pca_dims > original dims and is an int, ensure function handles gracefully (warns, no exception)
    if isinstance(pca_dims, int) and pca_dims and pca_dims > embeddings.shape[1]:
        # The implementation logs a warning and skips reduction; no exception expected.
        pass

    processed_embeddings, _ = post_process_embeddings(embeddings, pca_dims, sif_coefficient)

    # Assert the shape is correct
    assert processed_embeddings.shape == expected_shape

    # If SIF weighting is applied and no PCA reduction, check weights are applied correctly
    if sif_coefficient and pca_dims is None:
        inv_rank = 1 / (np.arange(2, embeddings.shape[0] + 2))
        proba = inv_rank / np.sum(inv_rank)
        sif_weights = (sif_coefficient / (sif_coefficient + proba))[:, None]

        expected_zipf_embeddings = original_embeddings * sif_weights
        assert np.allclose(processed_embeddings, expected_zipf_embeddings, rtol=1e-5), (
            "SIF weighting not applied correctly"
        )


@pytest.mark.parametrize(
    "added_tokens, expected_output, expected_warnings",
    [
        # Case: duplicates ("2010", "government") and an empty token ("")
        (["2010", "government", "nerv", ""], ["nerv"], ["Removed", "duplicate", "empty"]),
        # Case: No duplicates, no empty tokens
        (["worda", "wordb", "wordc"], ["worda", "wordb", "wordc"], []),
        # Case: Only empty token (""), should return an empty list
        ([""], [], ["Removed", "empty"]),
    ],
)
def test_clean_and_create_vocabulary(
    mock_berttokenizer: BertTokenizerFast,
    added_tokens: list[str],
    expected_output: list[str],
    expected_warnings: list[str],
    caplog: LogCaptureFixture,
) -> None:
    """Test the clean_and_create_vocabulary helper."""
    with caplog.at_level("WARNING"):
        tokens, _ = clean_and_create_vocabulary(mock_berttokenizer, added_tokens, None)

        cleaned_vocab = [token.form for token in tokens if not token.is_internal]
        # Check the cleaned vocabulary matches the expected output
        assert cleaned_vocab == expected_output

        # Check the warnings were logged as expected
        logged_warnings = [record.message for record in caplog.records]
        for expected_warning in expected_warnings:
            assert any(expected_warning in logged_warning for logged_warning in logged_warnings)


@pytest.mark.parametrize(
    "pooling,with_pooler,expected_rows",
    [
        (PoolingType.MEAN, False, [1.0, 0.0]),  # len=3: mean(0,1,2)=1; len=1: mean(0)=0
        (PoolingType.LAST, False, [2.0, 0.0]),  # last of 3: 2; last of 1: 0
        (PoolingType.FIRST, False, [0.0, 0.0]),  # first position: 0
        (PoolingType.POOLER, True, [7.0, 7.0]),  # pooler_output used
    ],
)
def test_pooling_strategies(mock_transformer, pooling, with_pooler, expected_rows) -> None:
    """Test different pooling strategies."""
    mock_transformer.with_pooler = with_pooler
    tokenized = [[10, 11, 12], [20]]
    out = create_embeddings(
        model=mock_transformer,
        tokenized=tokenized,
        device="cpu",
        pad_token_id=0,
        pooling=pooling,
    )
    dim = out.shape[1]
    expected = np.stack([np.full((dim,), v, dtype=np.float32) for v in expected_rows])
    assert np.allclose(out, expected, rtol=1e-6, atol=0.0)


def test_pooler_raises_without_pooler_output(mock_transformer) -> None:
    """POOLER should raise when the model doesn't expose pooler_output."""
    mock_transformer.with_pooler = False
    tokenized = [[10, 11, 12], [20]]
    with pytest.raises(ValueError, match="pooler_output"):
        _ = create_embeddings(
            model=mock_transformer,
            tokenized=tokenized,
            device="cpu",
            pad_token_id=0,
            pooling=PoolingType.POOLER,
        )
