from __future__ import annotations

import json
import re
from importlib import import_module
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import LogCaptureFixture
from transformers import AutoModel, BertTokenizerFast

from model2vec.distill.distillation import _clean_vocabulary, _post_process_embeddings, distill, distill_from_model
from model2vec.model import StaticModel

try:
    # For huggingface_hub>=0.25.0
    from huggingface_hub.errors import RepositoryNotFoundError
except ImportError:
    # For huggingface_hub<0.25.0
    from huggingface_hub.utils._errors import RepositoryNotFoundError

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
        (None, True, 0.9, True),  # PCA as float applied
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

    vocab_size = mock_berttokenizer.vocab_size

    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=None,
        device="cpu",
        token_remove_pattern=None,
    )

    assert len(static_model.embedding) == vocab_size

    # No tokens removed, nonsensical pattern
    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=None,
        device="cpu",
        token_remove_pattern="£££££££££££££££££",
    )

    assert len(static_model.embedding) == vocab_size

    with pytest.raises(ValueError):
        static_model = distill_from_model(
            model=mock_transformer,
            tokenizer=mock_berttokenizer,
            vocabulary=None,
            device="cpu",
            token_remove_pattern="[...papapa",
        )

    # Remove all tokens
    with pytest.raises(ValueError):
        static_model = distill_from_model(
            model=mock_transformer,
            tokenizer=mock_berttokenizer,
            vocabulary=None,
            device="cpu",
            token_remove_pattern=".*",
        )


@pytest.mark.parametrize(
    "vocabulary, use_subword, pca_dims, apply_zipf, sif_coefficient, expected_shape",
    [
        (None, True, 256, True, None, (29528, 256)),  # Output vocab with subwords, PCA applied
        (
            ["wordA", "wordB"],
            False,
            4,
            False,
            None,
            (7, 4),
        ),  # Custom vocab without subword , PCA applied
        (None, True, "auto", False, None, (29528, 768)),  # Subword, PCA set to 'auto'
        (None, True, "auto", True, 1e-4, (29528, 768)),  # Subword, PCA set to 'auto'
        (None, True, "auto", False, 1e-4, (29528, 768)),  # Subword, PCA set to 'auto'
        (None, True, "auto", True, 0, None),  # Sif too low
        (None, True, "auto", True, 1, None),  # Sif too high
        (None, True, "auto", False, 0, (29528, 768)),  # Sif too low, but apply_zipf is False
        (None, True, "auto", False, 1, (29528, 768)),  # Sif too high, but apply_zipf is False
        (None, True, 1024, False, None, (29528, 768)),  # Subword, PCA set to high number.
        (["wordA", "wordB"], True, 4, False, None, (29530, 4)),  # Custom vocab with subword, PCA applied
        (None, True, None, True, None, (29528, 768)),  # No PCA applied
        (["wordA", "wordB"], False, 4, True, None, (7, 4)),  # Custom vocab without subwords PCA and Zipf applied
        (None, False, 256, True, None, None),  # use_subword = False without passing a vocabulary should raise an error
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
    sif_coefficient: float | None,
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
                sif_coefficient=sif_coefficient,
            )
    elif (
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
                use_subword=use_subword,
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
            use_subword=use_subword,
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
            _post_process_embeddings(embeddings, pca_dims, None)

    processed_embeddings = _post_process_embeddings(embeddings, pca_dims, sif_coefficient)

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


def test_distill_with_bpe_whitespace_pretokenizer() -> None:
    """Test distill function with BPE tokenizer with whitespace pretokenizer and added vocabulary."""
    import json
    from unittest.mock import MagicMock
    from model2vec.distill.distillation import distill_from_model

    # Create a dummy BPE tokenizer with whitespace pretokenizer support.
    class DummyBPE:
        vocab_size = 100
        def __init__(self):
            self.vocab = {}
        def to_str(self) -> str:
            # Include the tokenizer type and current vocabulary in the JSON string.
            return json.dumps({"tokenizer": "BPE-Whitespace", "vocab": list(self.vocab.keys())})
    
    dummy_tokenizer = DummyBPE()
    dummy_model = MagicMock()
    
    extra_vocab = ["hello", "world"]
    # Simulate that the tokenizer adds extra vocabulary words.
    for word in extra_vocab:
        dummy_tokenizer.vocab[word] = len(dummy_tokenizer.vocab)
    
    # Call distill_from_model using the dummy BPE tokenizer with added vocabulary.
    static_model = distill_from_model(model=dummy_model, tokenizer=dummy_tokenizer, device="cpu", vocabulary=extra_vocab)
    
    tokens_str = static_model.tokenizer.to_str()
    # Verify that the tokenizer configuration from the BPE tokenizer is preserved.
    assert "BPE-Whitespace" in tokens_str
    # Check that the extra vocabulary words have been added to the tokenizer.
    for word in extra_vocab:
         assert word in tokens_str
