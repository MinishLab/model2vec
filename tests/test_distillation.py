from __future__ import annotations

import json
from importlib import import_module
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import LogCaptureFixture
from transformers import AutoModel, BertTokenizerFast

from model2vec.distill.distillation import (
    clean_and_create_vocabulary,
    distill,
    distill_from_model,
    post_process_embeddings,
)
from model2vec.model import StaticModel

try:
    # For huggingface_hub>=0.25.0
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    # For huggingface_hub<0.25.0
    from huggingface_hub.utils._errors import RepositoryNotFoundError  # type: ignore

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "vocabulary, pca_dims, apply_zipf",
    [
        (None, 256, True),  # Output vocab with subwords, PCA applied
        (["wordA", "wordB"], 4, False),  # Custom vocab with subword, PCA applied
        (None, "auto", False),  # Subword, PCA set to 'auto'
        (None, 1024, False),  # Subword, PCA set to high number.
        (None, None, True),  # No PCA applied
        (None, 0.9, True),  # PCA as float applied
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill_from_model(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_berttokenizer: BertTokenizerFast,
    mock_transformer: AutoModel,
    vocabulary: list[str] | None,
    pca_dims: int | None,
    apply_zipf: bool,
) -> None:
    """Test distill function with different parameters."""
    # Mock the return value of model_info to avoid calling the Hugging Face API
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})

    # Patch the tokenizers and models to return the real BertTokenizerFast and mock model instances
    # mock_auto_tokenizer.return_value = mock_berttokenizer
    mock_auto_model.return_value = mock_transformer

    # Call the distill function with the parametrized inputs
    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=vocabulary,
        device="cpu",
        pca_dims=pca_dims,
        apply_zipf=apply_zipf,
        token_remove_pattern=None,
    )

    static_model2 = distill(
        model_name="tests/data/test_tokenizer",
        vocabulary=vocabulary,
        device="cpu",
        pca_dims=pca_dims,
        apply_zipf=apply_zipf,
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
    mock_transformer: AutoModel,
) -> None:
    """Test the removal pattern."""
    # Mock the return value of model_info to avoid calling the Hugging Face API
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})

    # Patch the tokenizers and models to return the real BertTokenizerFast and mock model instances
    # mock_auto_tokenizer.return_value = mock_berttokenizer
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
        static_model = distill_from_model(
            model=mock_transformer,
            tokenizer=mock_berttokenizer,
            vocabulary=None,
            device="cpu",
            token_remove_pattern="[...papapa",
        )


@pytest.mark.parametrize(
    "vocabulary, pca_dims, apply_zipf, sif_coefficient, expected_shape",
    [
        (None, 256, True, None, (29524, 256)),  # Output vocab with subwords, PCA applied
        (None, "auto", False, None, (29524, 768)),  # Subword, PCA set to 'auto'
        (None, "auto", True, 1e-4, (29524, 768)),  # Subword, PCA set to 'auto'
        (None, "auto", False, 1e-4, (29524, 768)),  # Subword, PCA set to 'auto'
        (None, "auto", True, 0, None),  # Sif too low
        (None, "auto", True, 1, None),  # Sif too high
        (None, "auto", False, 0, (29524, 768)),  # Sif too low, but apply_zipf is False
        (None, "auto", False, 1, (29524, 768)),  # Sif too high, but apply_zipf is False
        (None, 1024, False, None, (29524, 768)),  # Subword, PCA set to high number.
        (["wordA", "wordB"], 4, False, None, (29526, 4)),  # Custom vocab with subword, PCA applied
        (None, None, True, None, (29524, 768)),  # No PCA applied
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_transformer: AutoModel,
    vocabulary: list[str] | None,
    pca_dims: int | None,
    apply_zipf: bool,
    sif_coefficient: float | None,
    expected_shape: tuple[int, int],
) -> None:
    """Test distill function with different parameters."""
    # Mock the return value of model_info to avoid calling the Hugging Face API
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})

    # Patch the tokenizers and models to return the real BertTokenizerFast and mock model instances
    mock_auto_model.return_value = mock_transformer

    model_name = "tests/data/test_tokenizer"

    if (
        apply_zipf is not None
        and apply_zipf
        and sif_coefficient is not None
        and (sif_coefficient <= 0 or sif_coefficient >= 1)
    ):
        with pytest.raises(ValueError):
            static_model = distill(
                model_name=model_name,
                vocabulary=vocabulary,
                device="cpu",
                pca_dims=pca_dims,
                apply_zipf=apply_zipf,
                sif_coefficient=sif_coefficient,
            )

    else:
        # Call the distill function with the parametrized inputs
        static_model = distill(
            model_name=model_name,
            vocabulary=vocabulary,
            device="cpu",
            pca_dims=pca_dims,
            apply_zipf=apply_zipf,
            sif_coefficient=sif_coefficient,
        )

        # Assert the model is correctly generated
        assert isinstance(static_model, StaticModel)
        assert static_model.embedding.shape == expected_shape
        assert "mock-model" in static_model.config["tokenizer_name"]
        assert static_model.tokenizer is not None


@patch.object(import_module("model2vec.distill.distillation"), "model_info")
def test_missing_modelinfo(
    mock_model_info: MagicMock,
    mock_transformer: AutoModel,
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
        (rng.random((1000, 768)), None, None, (1000, 768)),  # No PCA applied, dimensions remain unchanged
        (rng.random((1000, 768)), 256, 1e-4, (1000, 256)),  # PCA and Zipf applied
        (rng.random((10, 768)), 256, 1e-4, (10, 768)),  # PCA dims higher than vocab size, no PCA applied
    ],
)
def test__post_process_embeddings(
    embeddings: np.ndarray, pca_dims: int, sif_coefficient: float | None, expected_shape: tuple[int, int]
) -> None:
    """Test the _post_process_embeddings function."""
    original_embeddings = embeddings.copy()  # Copy embeddings to compare later

    # Test that the function raises an error if the PCA dims are larger than the number of dimensions
    if pca_dims and pca_dims > embeddings.shape[1]:
        with pytest.raises(ValueError):
            post_process_embeddings(embeddings, pca_dims, None)

    processed_embeddings = post_process_embeddings(embeddings, pca_dims, sif_coefficient)

    # Assert the shape is correct
    assert processed_embeddings.shape == expected_shape

    # If Zipf weighting is applied compare the original and processed embeddings
    # and check the weights are applied correctly
    if sif_coefficient and pca_dims is None:
        inv_rank = 1 / (np.arange(2, embeddings.shape[0] + 2))
        proba = inv_rank / np.sum(inv_rank)
        sif_weights = (sif_coefficient / (sif_coefficient + proba))[:, None]

        expected_zipf_embeddings = original_embeddings * sif_weights
        assert np.allclose(
            processed_embeddings, expected_zipf_embeddings, rtol=1e-5
        ), "Zipf weighting not applied correctly"


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
    """Test the _clean_vocabulary function."""
    with caplog.at_level("WARNING"):
        tokens, _ = clean_and_create_vocabulary(mock_berttokenizer, added_tokens, None)

        cleaned_vocab = [token.form for token in tokens if not token.is_internal]
        # Check the cleaned vocabulary matches the expected output
        assert cleaned_vocab == expected_output

        # Check the warnings were logged as expected
        logged_warnings = [record.message for record in caplog.records]

        # Ensure the expected warnings contain expected keywords like 'Removed', 'duplicate', or 'empty'
        for expected_warning in expected_warnings:
            assert any(expected_warning in logged_warning for logged_warning in logged_warnings)
