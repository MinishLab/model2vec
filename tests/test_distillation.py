# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import LogCaptureFixture
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import BertTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast

from model2vec.distill import init_from_tokenizer, random_init
from model2vec.distill.distillation import distill, distill_from_model
from model2vec.distill.inference import PoolingMode, apply_pca, compute_weights, create_embeddings
from model2vec.model import StaticModel
from model2vec.tokenizer import clean_and_create_vocabulary, turn_tokens_into_ids

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
        (["star wars"], 8, None),  # Multiword vocabulary
        (["..."], 8, None),  # Crashing multiword vocabulary
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill_from_model(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_berttokenizer: PreTrainedTokenizerFast,
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

    for token in vocabulary or []:
        # Normalized tokens are for single-word tokens.
        # Other tokens are added as addedtokens, as is.
        normalized = static_model.tokenizer.normalizer.normalize_str(token)
        assert token in static_model.tokens or normalized in static_model.tokens


@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill_quantization(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_berttokenizer: PreTrainedTokenizerFast,
    mock_transformer: PreTrainedModel,
) -> None:
    """Test distill function with different parameters."""
    # Mock the return value of model_info to avoid calling the Hugging Face API
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})
    mock_auto_model.return_value = mock_transformer

    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=None,
        device="cpu",
        pca_dims="auto",
        sif_coefficient=1e-4,
        token_remove_pattern=None,
        vocabulary_quantization=3,
    )

    assert static_model.embedding.shape == (3, 768)
    assert static_model.weights is not None
    assert static_model.token_mapping is not None
    assert len(static_model.weights) == static_model.tokenizer.get_vocab_size()


@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
def test_distill_removal_pattern_all_tokens(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_berttokenizer: BertTokenizer,
    mock_transformer: PreTrainedModel,
) -> None:
    """Test the removal pattern."""
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})
    mock_auto_model.return_value = mock_transformer

    # Even if we remove all tokens, we can't remove the [UNK] token
    model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=None,
        device="cpu",
        token_remove_pattern=r".*",
    )

    # So the only token left is the [UNK] token.
    assert model.tokens == ("[UNK]",)


@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
@pytest.mark.parametrize("mock_transformer", [{"vocab_size": 35022}], indirect=True)
def test_distill_removal_pattern(
    mock_auto_model: MagicMock,
    mock_model_info: MagicMock,
    mock_berttokenizer: PreTrainedTokenizerFast,
    mock_transformer: PreTrainedModel,
) -> None:
    """Test the removal pattern."""
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": {"language": "en"}})
    mock_auto_model.return_value = mock_transformer

    # Because the added [MASK], [CLS] and [SEP] get removed
    expected_vocab_size = mock_berttokenizer.vocab_size - 3

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

    # Test whether regexes remove words from the vocabulary
    static_model = distill_from_model(
        model=mock_transformer,
        tokenizer=mock_berttokenizer,
        vocabulary=["hellooooooo"],
        device="cpu",
        token_remove_pattern="hellooooooo",
    )
    assert "hellooooooo" not in static_model.tokens

    # Weird pattern.
    with pytest.raises(ValueError):
        _ = distill_from_model(
            model=mock_transformer,
            tokenizer=mock_berttokenizer,
            vocabulary=None,
            device="cpu",
            token_remove_pattern="[...papapa",
        )

    # Random init from the tokenizer only
    full = init_from_tokenizer(tokenizer=mock_berttokenizer, dim=8, token_remove_pattern=None)
    pruned = init_from_tokenizer(tokenizer=mock_berttokenizer, dim=8)
    assert len(full.embedding) == expected_vocab_size
    assert any(token.startswith("[unused") for token in full.tokens)
    assert not any(token.startswith("[unused") for token in pruned.tokens)
    assert len(pruned.embedding) == len(pruned.tokens)


@pytest.mark.parametrize(
    "vocabulary, pca_dims, sif_coefficient, expected_shape",
    [
        (None, 256, None, (30519, 256)),  # PCA applied, SIF off
        (None, "auto", None, (30519, 768)),  # PCA 'auto', SIF off
        (None, "auto", 1e-4, (30519, 768)),  # PCA 'auto', SIF on
        (None, "auto", 0, None),  # invalid SIF (too low) -> raises
        (None, "auto", 1, None),  # invalid SIF (too high) -> raises
        (None, 1024, None, (30519, 768)),  # PCA set high (no reduction)
        (["wordA", "wordB"], 4, None, (30521, 4)),  # Custom vocab, PCA applied
        (None, None, None, (30519, 768)),  # No PCA, SIF off
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
@patch("transformers.AutoModel.from_pretrained")
@pytest.mark.parametrize("mock_transformer", [{"vocab_size": 30522}], indirect=True)
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
                token_remove_pattern=None,
            )
    else:
        static_model = distill(
            model_name=model_name,
            vocabulary=vocabulary,
            device="cpu",
            pca_dims=pca_dims,
            sif_coefficient=sif_coefficient,
            token_remove_pattern=None,
        )
        assert isinstance(static_model, StaticModel)
        assert static_model.embedding.shape == expected_shape
        assert static_model.embedding.flags["C_CONTIGUOUS"]
        assert "mock-model" in static_model.config["tokenizer_name"]
        assert static_model.tokenizer is not None


@patch.object(import_module("model2vec.distill.distillation"), "model_info")
def test_missing_modelinfo(
    mock_model_info: MagicMock,
    mock_transformer: PreTrainedModel,
    mock_berttokenizer: PreTrainedTokenizerFast,
) -> None:
    """Test that missing model info does not crash."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_model_info.side_effect = RepositoryNotFoundError("Model not found", response=mock_response)
    static_model = distill_from_model(model=mock_transformer, tokenizer=mock_berttokenizer, device="cpu")
    assert static_model.language is None


@pytest.mark.parametrize(
    "vocabulary, dim, sif_coefficient, token_remove_pattern, quantize_to, expected_shape",
    [
        (None, 256, 1e-4, None, "float16", (30519, 256)),  # SIF on
        (None, 16, None, None, "float32", (30519, 16)),  # SIF off
        (None, 16, None, None, "float64", (30519, 16)),
        (["wordA", "wordB"], 8, None, None, "int8", (30521, 8)),  # Custom vocab
        (None, 0, None, None, "float16", None),  # invalid dim -> raises
        (None, -1, None, None, "float16", None),  # invalid dim -> raises
        (None, 8, 0, None, "float16", None),  # invalid SIF (too low) -> raises
        (None, 8, 1, None, "float16", None),  # invalid SIF (too high) -> raises
        (None, 8, None, "[", "float16", None),  # invalid regex -> raises
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
def test_init_from_tokenizer(
    mock_model_info: MagicMock,
    mock_berttokenizer: PreTrainedTokenizerFast,
    tmp_path: Path,
    vocabulary: list[str] | None,
    dim: int,
    sif_coefficient: float | None,
    token_remove_pattern: str | None,
    quantize_to: str,
    expected_shape: tuple[int, int] | None,
) -> None:
    """Test creating a random model from a tokenizer only."""
    if expected_shape is None:
        with pytest.raises(ValueError):
            init_from_tokenizer(
                tokenizer=mock_berttokenizer,
                dim=dim,
                sif_coefficient=sif_coefficient,
                token_remove_pattern=token_remove_pattern,
            )
        return

    static_model = init_from_tokenizer(
        tokenizer=mock_berttokenizer,
        vocabulary=vocabulary,
        dim=dim,
        sif_coefficient=sif_coefficient,
        token_remove_pattern=token_remove_pattern,
        quantize_to=quantize_to,
    )
    assert static_model.embedding.shape == expected_shape
    assert static_model.embedding.dtype == np.dtype(quantize_to)
    assert static_model.weights is None
    assert static_model.token_mapping is None
    assert static_model.config["hidden_dim"] == dim
    assert static_model.config["apply_pca"] is None
    assert static_model.config["sif_coefficient"] == sif_coefficient
    assert static_model.config["pooling"] == PoolingMode.RANDOM
    assert static_model.base_model_name == "test_tokenizer"
    mock_model_info.assert_not_called()

    for token in vocabulary or []:
        assert static_model.tokenizer.normalizer.normalize_str(token) in static_model.tokens
    assert static_model.encode([" ".join(vocabulary or ["hello"])]).shape == (1, dim)

    static_model.save_pretrained(tmp_path)
    loaded = StaticModel.from_pretrained(tmp_path)
    assert np.array_equal(loaded.embedding, static_model.embedding)
    assert loaded.tokens == static_model.tokens

    seeded = init_from_tokenizer(tokenizer=mock_berttokenizer, dim=dim, seed=1)
    seeded_again = init_from_tokenizer(tokenizer=mock_berttokenizer, dim=dim, seed=1)
    other_seed = init_from_tokenizer(tokenizer=mock_berttokenizer, dim=dim, seed=2)
    assert np.array_equal(seeded.embedding, seeded_again.embedding)
    assert not np.array_equal(seeded.embedding, other_seed.embedding)


@pytest.fixture
def byte_level_tokenizer() -> PreTrainedTokenizerFast:
    """Create a byte-level BPE tokenizer without a prefix space."""
    vocab = ["word1", "word2", "Ġword3", "[UNK]"]
    tokenizer = Tokenizer(BPE(vocab={token: idx for idx, token in enumerate(vocab)}, merges=[], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)  # type: ignore
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


def test_init_from_tokenizer_byte_level(byte_level_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test that byte-level tokenizers get a prefix space, and that removing all tokens raises."""
    static_model = init_from_tokenizer(tokenizer=byte_level_tokenizer, dim=4, token_remove_pattern=None)
    assert static_model.tokenizer.pre_tokenizer.add_prefix_space  # type: ignore
    assert not byte_level_tokenizer.backend_tokenizer.pre_tokenizer.add_prefix_space  # type: ignore
    assert static_model.embedding.shape == (4, 4)

    with pytest.raises(ValueError, match="The vocabulary is empty after preprocessing"):
        init_from_tokenizer(tokenizer=byte_level_tokenizer, dim=4, token_remove_pattern=r".*")


def test_init_from_tokenizer_sif_weighting(mock_berttokenizer: PreTrainedTokenizerFast) -> None:
    """Test that SIF weights are folded into the embeddings."""
    unweighted = init_from_tokenizer(
        tokenizer=mock_berttokenizer, dim=8, sif_coefficient=None, token_remove_pattern=None, quantize_to="float64"
    )
    weighted = init_from_tokenizer(
        tokenizer=mock_berttokenizer, dim=8, sif_coefficient=1e-4, token_remove_pattern=None, quantize_to="float64"
    )
    weights = compute_weights(len(unweighted.embedding), sif_coefficient=1e-4)
    assert np.allclose(weighted.embedding, unweighted.embedding * weights[:, None])


@pytest.mark.parametrize(
    "name_or_path, card_data, expected_language",
    [
        ("", None, None),  # No name -> no hub call
        ("some-org/some-model", {"language": "en"}, "en"),  # Hub name -> language from model card
    ],
)
@patch.object(import_module("model2vec.distill.distillation"), "model_info")
def test_init_from_tokenizer_model_info(
    mock_model_info: MagicMock,
    mock_berttokenizer: PreTrainedTokenizerFast,
    name_or_path: str,
    card_data: dict[str, str] | None,
    expected_language: str | None,
) -> None:
    """Test that the hub is only queried for tokenizers with a hub name."""
    mock_model_info.return_value = type("ModelInfo", (object,), {"cardData": card_data})
    mock_berttokenizer.name_or_path = name_or_path
    static_model = init_from_tokenizer(tokenizer=mock_berttokenizer, dim=8)
    assert static_model.language == expected_language
    assert static_model.config["tokenizer_name"] == name_or_path
    if card_data is None:
        mock_model_info.assert_not_called()
    else:
        mock_model_info.assert_called_once_with(name_or_path)
        assert static_model.base_model_name == name_or_path


@patch("model2vec.distill.distillation.AutoTokenizer.from_pretrained")
def test_random_init(mock_from_pretrained: MagicMock, mock_berttokenizer: PreTrainedTokenizerFast) -> None:
    """Test that random_init loads the tokenizer and matches init_from_tokenizer."""
    mock_from_pretrained.return_value = mock_berttokenizer
    static_model = random_init("tests/data/test_tokenizer", dim=8, seed=3, trust_remote_code=True)
    mock_from_pretrained.assert_called_once_with("tests/data/test_tokenizer", trust_remote_code=True, use_fast=True)
    expected = init_from_tokenizer(tokenizer=mock_berttokenizer, dim=8, seed=3)
    assert np.array_equal(static_model.embedding, expected.embedding)
    assert static_model.tokens == expected.tokens


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

    processed_embeddings = apply_pca(embeddings, pca_dims)
    weights = compute_weights(len(processed_embeddings), sif_coefficient=sif_coefficient)
    processed_embeddings = processed_embeddings * weights[:, None]

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
        (["2010", "government", "nerv", ""], ["nerv"], ["already", "empty"]),
        # Case: No duplicates, no empty tokens
        (["worda", "wordb", "wordc"], ["worda", "wordb", "wordc"], []),
        # Case: Only empty token (""), should return an empty list
        ([""], [], ["empty"]),
        (["multi word token"], ["multi word token"], []),
        # Case: duplicate multi-word token, should only be added once
        (["multi word token", "multi word token"], ["multi word token"], ["twice"]),
        # Case: multi-word tokens that normalize to the same string, should only be added once
        (["Multi Word Token", "multi word token"], ["multi word token"], ["twice"]),
    ],
)
def test_clean_and_create_vocabulary(
    mock_tokenizermodel: TokenizerModel,
    added_tokens: list[str],
    expected_output: list[str],
    expected_warnings: list[str],
    caplog: LogCaptureFixture,
) -> None:
    """Test the clean_and_create_vocabulary helper."""
    with caplog.at_level("WARNING"):
        old_tokens = mock_tokenizermodel.sorted_vocabulary
        tokenizer_model = clean_and_create_vocabulary(mock_tokenizermodel, added_tokens, None)
        tokens = set(tokenizer_model.sorted_vocabulary) - set(old_tokens)

        # Check the cleaned vocabulary matches the expected output
        assert tokens == set(expected_output)

        # Check the warnings were logged as expected
        logged_warnings = [record.message for record in caplog.records]
        for expected_warning in expected_warnings:
            assert any(expected_warning in logged_warning for logged_warning in logged_warnings)


@pytest.mark.parametrize(
    "token,expected_tokens",
    [
        ("##ing", ["[CLS]", "##ing", "[SEP]"]),
        ("##xyzq", ["[CLS]", "x", "##y", "##z", "##q", "[SEP]"]),
        ("hello world", ["[CLS]", "hello", "world", "[SEP]"]),
    ],
)
def test_turn_tokens_into_ids(mock_tokenizermodel: TokenizerModel, token: str, expected_tokens: list[str]) -> None:
    """Test that tokens are decoded before being re-encoded."""
    ids = turn_tokens_into_ids([token], mock_tokenizermodel)
    expected_ids = [mock_tokenizermodel.vocabulary[t] for t in expected_tokens]
    assert ids == [expected_ids]


def test_turn_tokens_into_ids_byte_level() -> None:
    """Test that unmatched byte-level tokens are decoded to text with a leading space before being re-encoded."""
    vocab = ["[UNK]", "Ġ", "Ä", "ł", "w", "o", "r", "d", "1"]
    tokenizer = Tokenizer(BPE(vocab={token: idx for idx, token in enumerate(vocab)}, merges=[], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)  # type: ignore
    model = TokenizerModel.from_transformers_tokenizer(PreTrainedTokenizerFast(tokenizer_object=tokenizer))
    assert "Ġword1" not in model.vocabulary

    ids = turn_tokens_into_ids(["Ġword1"], model)
    expected_ids = model.to_tokenizer().encode(" word1").ids
    assert expected_ids == [model.vocabulary[t] for t in ["Ġ", "w", "o", "r", "d", "1"]]
    assert ids == [expected_ids]


@pytest.mark.parametrize(
    "pooling,with_pooler,expected_rows",
    [
        (PoolingMode.MEAN, False, [1.0, 0.0]),  # len=3: mean(0,1,2)=1; len=1: mean(0)=0
        (PoolingMode.LAST, False, [2.0, 0.0]),  # last of 3: 2; last of 1: 0
        (PoolingMode.FIRST, False, [0.0, 0.0]),  # first position: 0
        (PoolingMode.POOLER, True, [7.0, 7.0]),  # pooler_output used
    ],
)
def test_pooling_strategies(
    mock_transformer: PreTrainedModel, pooling: PoolingMode, with_pooler: bool, expected_rows: tuple[float, float]
) -> None:
    """Test different pooling strategies."""
    mock_transformer.with_pooler = with_pooler  # type: ignore
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


def test_pooler_raises_without_pooler_output(mock_transformer: PreTrainedModel) -> None:
    """POOLER should raise when the model doesn't expose pooler_output."""
    mock_transformer.with_pooler = False  # type: ignore
    tokenized = [[10, 11, 12], [20]]
    with pytest.raises(ValueError, match="pooler_output"):
        _ = create_embeddings(
            model=mock_transformer,
            tokenized=tokenized,
            device="cpu",
            pad_token_id=0,
            pooling=PoolingMode.POOLER,
        )
