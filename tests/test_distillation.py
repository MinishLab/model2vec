import json
from importlib import import_module
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from huggingface_hub.utils._errors import RepositoryNotFoundError
from pytest import LogCaptureFixture
from transformers import AutoModel, BertTokenizerFast

from model2vec.distill.distillation import _clean_vocabulary, _post_process_embeddings, distill, distill_from_model
from model2vec.model import StaticModel

rng = np.random.default_rng()


@pytest.mark.parametrize(
    "vocabulary, use_subword, pca_dims, apply_zipf",
    [
        (None, True, 256, True),  # Output vocab with subwords, PCA applied
        (
            ["wordA", "wordB"],
            False,
            4,
            False,
        ),  # Custom vocab without subword , PCA applied
        (["wordA", "wordB"], True, 4, False),  # Custom vocab with subword, PCA applied
        (None, True, "auto", False),  # Subword, PCA set to 'auto'
        (None, True, 1024, False),  # Subword, PCA set to high number.
        (None, True, None, True),  # No PCA applied
        (["wordA", "wordB"], False, 4, True),  # Custom vocab without subwords PCA and Zipf applied
        (None, False, 256, True),  # use_subword = False without passing a vocabulary should raise an error
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
    use_subword: bool,
    pca_dims: int | None,
    apply_zipf: bool,
) -> None:
    """Test distill function with different parameters."""
    # Mock the return value of model_info to avoid calling the Hugging Face API
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})

    # Patch the tokenizers and models to return the real BertTokenizerFast and mock model instances
    # mock_auto_tokenizer.return_value = mock_berttokenizer
    mock_auto_model.return_value = mock_transformer

    if vocabulary is None and not use_subword:
        with pytest.raises(ValueError):
            static_model = distill_from_model(
                model=mock_transformer,
                tokenizer=mock_berttokenizer,
                vocabulary=vocabulary,
                device="cpu",
                pca_dims=pca_dims,
                apply_zipf=apply_zipf,
                use_subword=use_subword,
            )
    else:
        # Call the distill function with the parametrized inputs
        static_model = distill_from_model(
            model=mock_transformer,
            tokenizer=mock_berttokenizer,
            vocabulary=vocabulary,
            device="cpu",
            pca_dims=pca_dims,
            apply_zipf=apply_zipf,
            use_subword=use_subword,
        )

        static_model2 = distill(
            model_name="tests/data/test_tokenizer",
            vocabulary=vocabulary,
            device="cpu",
            pca_dims=pca_dims,
            apply_zipf=apply_zipf,
            use_subword=use_subword,
        )

        assert static_model.embedding.weight.shape == static_model2.embedding.weight.shape
        assert static_model.config == static_model2.config
        assert json.loads(static_model.tokenizer.to_str()) == json.loads(static_model2.tokenizer.to_str())
        assert static_model.base_model_name == static_model2.base_model_name


@pytest.mark.parametrize(
    "vocabulary, use_subword, pca_dims, apply_zipf, expected_shape",
    [
        (None, True, 256, True, (29528, 256)),  # Output vocab with subwords, PCA applied
        (
            ["wordA", "wordB"],
            False,
            4,
            False,
            (7, 4),
        ),  # Custom vocab without subword , PCA applied
        (None, True, "auto", False, (29528, 768)),  # Subword, PCA set to 'auto'
        (None, True, 1024, False, (29528, 768)),  # Subword, PCA set to high number.
        (["wordA", "wordB"], True, 4, False, (29530, 4)),  # Custom vocab with subword, PCA applied
        (None, True, None, True, (29528, 768)),  # No PCA applied
        (["wordA", "wordB"], False, 4, True, (7, 4)),  # Custom vocab without subwords PCA and Zipf applied
        (None, False, 256, True, None),  # use_subword = False without passing a vocabulary should raise an error
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_transformer: AutoModel,
    vocabulary: list[str] | None,
    use_subword: bool,
    pca_dims: int | None,
    apply_zipf: bool,
    expected_shape: tuple[int, int],
) -> None:
    """Test distill function with different parameters."""
    # Mock the return value of model_info to avoid calling the Hugging Face API
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})

    # Patch the tokenizers and models to return the real BertTokenizerFast and mock model instances
    mock_auto_model.return_value = mock_transformer

    model_name = "tests/data/test_tokenizer"

    if vocabulary is None and not use_subword:
        with pytest.raises(ValueError):
            static_model = distill(
                model_name=model_name,
                vocabulary=vocabulary,
                device="cpu",
                pca_dims=pca_dims,
                apply_zipf=apply_zipf,
                use_subword=use_subword,
            )
    else:
        # Call the distill function with the parametrized inputs
        static_model = distill(
            model_name=model_name,
            vocabulary=vocabulary,
            device="cpu",
            pca_dims=pca_dims,
            apply_zipf=apply_zipf,
            use_subword=use_subword,
        )

        # Assert the model is correctly generated
        assert isinstance(static_model, StaticModel)
        assert static_model.embedding.weight.shape == expected_shape
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
    "embeddings, pca_dims, apply_zipf, expected_shape",
    [
        (rng.random((1000, 768)), 256, False, (1000, 256)),  # PCA applied correctly
        (rng.random((1000, 768)), None, False, (1000, 768)),  # No PCA applied, dimensions remain unchanged
        (rng.random((1000, 768)), 256, True, (1000, 256)),  # PCA and Zipf applied
        (rng.random((10, 768)), 256, False, (10, 768)),  # PCA dims higher than vocab size, no PCA applied
    ],
)
def test__post_process_embeddings(
    embeddings: np.ndarray, pca_dims: int, apply_zipf: bool, expected_shape: tuple[int, int]
) -> None:
    """Test the _post_process_embeddings function."""
    original_embeddings = embeddings.copy()  # Copy embeddings to compare later

    # Test that the function raises an error if the PCA dims are larger than the number of dimensions
    if pca_dims and pca_dims > embeddings.shape[1]:
        with pytest.raises(ValueError):
            _post_process_embeddings(embeddings, pca_dims, False)

    processed_embeddings = _post_process_embeddings(embeddings, pca_dims, apply_zipf)

    # Assert the shape is correct
    assert processed_embeddings.shape == expected_shape

    # If Zipf weighting is applied compare the original and processed embeddings
    # and check the weights are applied correctly
    if apply_zipf and pca_dims is None:
        zipf_weights = np.log(1 + np.arange(embeddings.shape[0]))[:, None]
        expected_zipf_embeddings = original_embeddings * zipf_weights
        assert np.allclose(
            processed_embeddings, expected_zipf_embeddings, rtol=1e-5
        ), "Zipf weighting not applied correctly"


@pytest.mark.parametrize(
    "preprocessed_vocabulary, added_tokens, expected_output, expected_warnings",
    [
        # Case: duplicates ("word2") and an empty token ("")
        (["word1", "word2", "word2", "word3", ""], ["word3"], ["word1", "word2"], ["Removed", "duplicate", "empty"]),
        # Case: No duplicates, no empty tokens
        (["wordA", "wordB", "wordC"], [], ["wordA", "wordB", "wordC"], []),
        # Case: Duplicate "wordB" and "wordA" already in added_tokens
        (["wordA", "wordB", "wordC", "wordB"], ["wordA"], ["wordB", "wordC"], ["Removed", "duplicate"]),
        # Case: Only empty token (""), should return an empty list
        ([""], [], [], ["Removed", "empty"]),
    ],
)
def test__clean_vocabulary(
    preprocessed_vocabulary: list[str],
    added_tokens: list[str],
    expected_output: list[str],
    expected_warnings: list[str],
    caplog: LogCaptureFixture,
) -> None:
    """Test the _clean_vocabulary function."""
    with caplog.at_level("WARNING"):
        cleaned_vocab = _clean_vocabulary(preprocessed_vocabulary, added_tokens)

        # Check the cleaned vocabulary matches the expected output
        assert cleaned_vocab == expected_output

        # Check the warnings were logged as expected
        logged_warnings = [record.message for record in caplog.records]

        # Ensure the expected warnings contain expected keywords like 'Removed', 'duplicate', or 'empty'
        for expected_warning in expected_warnings:
            assert any(expected_warning in logged_warning for logged_warning in logged_warnings)
