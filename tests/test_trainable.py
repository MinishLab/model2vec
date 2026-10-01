import logging
import pickle
from collections import Counter, UserList
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from model2vec.model import StaticModel
from model2vec.train import StaticModelForClassification
from model2vec.train.base import BaseFinetuneable
from model2vec.train.dataset import (
    ClassTargets,
    ColumnRows,
    PairDataset,
    TextDataset,
    read_column,
)
from model2vec.train.pairs import PairInfoNCELoss, StaticModelForPairSimilarity
from model2vec.train.regression import StaticModelForRegression
from model2vec.train.similarity import StaticModelForSimilarity
from model2vec.train.trainer import _resolve_max_epochs, resolve_device, run_training_loop
from model2vec.train.utils import (
    get_probable_pad_token_id,
    logit,
    seed_everything,
    split_indices,
)


@pytest.mark.parametrize("n_layers", [0, 1, 2, 3])
def test_init_predict(n_layers: int, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test successful initialization of StaticModelForClassification."""
    vectors_torched = torch.from_numpy(mock_vectors)
    s = StaticModelForClassification(vectors=vectors_torched, tokenizer=mock_tokenizer, n_layers=n_layers)
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]
    assert list(s.classes) == s.classes_
    assert list(s.classes) == ["0", "1"]

    head = s.construct_head()
    assert head[0].in_features == mock_vectors.shape[1]
    head = s.construct_head()
    assert head[0].in_features == mock_vectors.shape[1]
    assert head[-1].out_features == 2


def test_init_base_class(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test successful initialization of the base class."""
    vectors_torched = torch.from_numpy(mock_vectors)
    s = BaseFinetuneable(
        vectors=vectors_torched, tokenizer=mock_tokenizer, hidden_dim=256, out_dim=3, n_layers=0, pad_id=0
    )
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]

    head = s.construct_head()
    assert head[0].in_features == mock_vectors.shape[1]


def test_trainable_tokenizer_does_not_pad(mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity) -> None:
    """The tokenizer of a trainable model keeps its pad token, but doesn't pad."""
    model = mock_trained_pair_similarity_pipeline
    assert model.tokenizer.padding is not None
    assert model._batch_tokenizer(None)(["word1 word2", "word2"])[1] == model._batch_tokenizer(None)(["word2"])[0]


def test_empty_texts_have_finite_gradients(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Texts without any tokens encode to zero vectors and don't produce NaN gradients."""
    torch.manual_seed(0)
    model = StaticModelForClassification(
        vectors=torch.from_numpy(mock_vectors).float() * 1e20, tokenizer=mock_tokenizer, n_layers=0
    )
    dataset = model._text_dataset(ColumnRows(text=["word1 word2", ""], label=["0", "1"]))
    batch, y = next(iter(dataset.to_dataloader(shuffle=False, batch_size=2)))

    nn.functional.cross_entropy(model(batch), y).backward()

    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_init_base_from_model(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test initializion from a static model."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    s = BaseFinetuneable.from_static_model(model=model)
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]

    with TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        s = BaseFinetuneable.from_pretrained(model_name=temp_dir)
        assert s.vectors.shape == mock_vectors.shape
        assert s.w.shape[0] == mock_vectors.shape[0]


def test_init_classifier_from_model(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test initializion from a static model."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    s = StaticModelForClassification.from_static_model(model=model)
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]

    with TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        s = StaticModelForClassification.from_pretrained(model_name=temp_dir)
        assert s.vectors.shape == mock_vectors.shape
        assert s.w.shape[0] == mock_vectors.shape[0]


def test_init_classifier_from_model_w(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test initializion from a static model."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, weights=np.ones(len(mock_vectors)))
    s = StaticModelForClassification.from_static_model(model=model)
    assert s._weights is not None
    assert torch.all(s._weights == torch.ones(len(mock_vectors)))
    w = s.construct_weights()
    assert w.shape[0] == mock_vectors.shape[0]
    assert torch.all(w == torch.ones(len(mock_vectors)))


def test_pad_token(mock_tokenizer: Tokenizer) -> None:
    """Test initializion from a static model."""
    tokenizer_model = TokenizerModel.from_tokenizer(mock_tokenizer)
    tokenizer_model.pad_token = "[HELLO]"
    tokenizer = tokenizer_model.to_tokenizer()
    vectors = np.random.RandomState().randn(6, 10)
    model = StaticModel(vectors=vectors, tokenizer=tokenizer)
    s = StaticModelForClassification.from_static_model(model=model, pad_token="[HELLO]")
    assert s.w.shape[0] == vectors.shape[0]
    assert s.pad_id == 5

    with pytest.raises(KeyError):
        StaticModelForClassification.from_static_model(model=model, pad_token="[BRR]")


def test_encode(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the encode function."""
    result = mock_trained_pipeline._encode(torch.tensor([[0, 1], [1, 0]]).long())
    assert result.shape == (2, 12)
    assert torch.allclose(result[0], result[1])


def test_tokenize(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the encode function."""
    result = mock_trained_pipeline.tokenize(["dog dog", "cat"])
    assert result.shape == torch.Size([2, 2])
    assert result[1, 1] == 0


def test_device(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Get the device."""
    assert mock_trained_pipeline.device == torch.device(type="cpu")  # type: ignore  # False positive
    assert mock_trained_pipeline.device == mock_trained_pipeline.w.device


def test_conversion(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the conversion to numpy."""
    staticmodel = mock_trained_pipeline.to_static_model()
    with torch.no_grad():
        result_1 = mock_trained_pipeline._encode(torch.tensor([[1, 2], [2, 1]]).long()).numpy()
    result_2 = staticmodel.embedding[[[1, 2], [2, 1]]].mean(0)
    result_2 /= np.linalg.norm(result_2, axis=1, keepdims=True)

    assert np.allclose(result_1, result_2)


def test_token_dropout_default_is_zero(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """A freshly constructed model has token dropout disabled."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    assert s.token_dropout == 0.0


def test_apply_token_dropout_noop_in_eval(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Token dropout has no effect in eval mode, even with a high dropout rate."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    s.token_dropout = 0.9
    s.eval()
    mask = torch.ones(4, 5)
    assert torch.equal(s._apply_token_dropout(mask), mask)


def test_apply_token_dropout_noop_when_rate_is_zero(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Token dropout has no effect when the rate is 0, even in training mode."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    s.train()
    s.token_dropout = 0.0
    mask = torch.ones(4, 5)
    assert torch.equal(s._apply_token_dropout(mask), mask)


def test_apply_token_dropout_never_empties_a_nonempty_row(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Even at a very high dropout rate, a row with at least one real token keeps at least one."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    s.train()
    s.token_dropout = 0.99
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])

    torch.manual_seed(0)
    for _ in range(20):
        out = s._apply_token_dropout(mask)
        assert (out.sum(dim=1) >= 1).all()


def test_apply_token_dropout_leaves_fully_padded_rows_untouched(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """A row with no real tokens to begin with is not force-filled by the dropout rescue."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    s.train()
    s.token_dropout = 0.99
    mask = torch.zeros(1, 4)

    torch.manual_seed(0)
    for _ in range(20):
        out = s._apply_token_dropout(mask)
        assert out.sum() == 0


def test_apply_token_dropout_drops_some_tokens(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """A moderate dropout rate actually zeroes out some, but not all, tokens over enough trials."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    s.train()
    s.token_dropout = 0.5
    mask = torch.ones(1, 1000)

    torch.manual_seed(0)
    out = s._apply_token_dropout(mask)
    assert 0 < out.sum().item() < mask.sum().item()


def test_fit_invalid_token_dropout_raises() -> None:
    """token_dropout must lie in [0, 1); out-of-range values raise ValueError."""
    tokenizer = AutoTokenizer.from_pretrained("tests/data/test_tokenizer").backend_tokenizer
    torch.random.manual_seed(42)
    vectors_torched = torch.randn(len(tokenizer.get_vocab()), 12)
    model = StaticModelForClassification(vectors=vectors_torched, tokenizer=tokenizer, hidden_dim=12).to("cpu")

    X = ["dog", "cat"]
    y = ["0", "1"]

    with pytest.raises(ValueError):
        model.fit(X, y, token_dropout=1.0, max_epochs=1)
    with pytest.raises(ValueError):
        model.fit(X, y, token_dropout=-0.1, max_epochs=1)


def test_fit_sets_token_dropout_and_disables_it_after_training() -> None:
    """fit() stores the requested token_dropout, and training ends in eval mode so it stops applying."""
    tokenizer = AutoTokenizer.from_pretrained("tests/data/test_tokenizer").backend_tokenizer
    torch.random.manual_seed(42)
    vectors_torched = torch.randn(len(tokenizer.get_vocab()), 12)
    model = StaticModelForClassification(vectors=vectors_torched, tokenizer=tokenizer, hidden_dim=12).to("cpu")

    X = ["dog", "cat"]
    y = ["0", "1"]

    model.fit(X, y, token_dropout=0.3, max_epochs=2, early_stopping_patience=1)

    assert model.token_dropout == 0.3
    assert model.training is False

    tokens = model.tokenize(["dog cat", "dog"])
    with torch.no_grad():
        first = model._encode(tokens)
        second = model._encode(tokens)
    assert torch.allclose(first, second)


def _pretokenized(texts: list[Any]) -> list[list[int]]:
    return list(texts)


def test_textdataset_init() -> None:
    """A text dataset has one item per row, or per index if indices are given."""
    rows = ColumnRows(text=[[1], [2], [3]], label=torch.arange(3))
    assert len(TextDataset(rows, _pretokenized, torch.as_tensor)) == 3
    dataset = TextDataset(rows, _pretokenized, torch.as_tensor, indices=np.array([2, 0]))
    assert len(dataset) == 2
    assert dataset[0][0] == [3]
    assert dataset[0][1].item() == 2


def test_column_rows() -> None:
    """Rows are fetched per column, and tensors are indexed along their first dimension."""
    rows = ColumnRows(text=["a", "b", "c"], label=torch.arange(6).reshape(3, 2))
    assert len(rows) == 3
    fetched = rows[[2, 0]]
    assert fetched["text"] == ["c", "a"]
    assert fetched["label"].tolist() == [[4, 5], [0, 1]]
    with pytest.raises(ValueError):
        ColumnRows(text=["a"], label=torch.arange(2))


def test_training_batch_padding_is_masked(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Training batches should pad with the model's pad id, so padding stays masked and out of the mean."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer, pad_id=1)
    texts = ["word2", "word2 word3"]

    dataset = s._text_dataset(ColumnRows(text=texts, label=["0", "1"]))
    batch, _ = next(iter(dataset.to_dataloader(shuffle=False, batch_size=2)))

    assert torch.equal(batch, s.tokenize(texts))
    with torch.no_grad():
        assert torch.allclose(s._encode(batch)[0], s._encode(s.tokenize(texts[:1]))[0])


def test_unknown_tokens_are_dropped(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Training and inference should drop unknown tokens, the way `StaticModel.tokenize` does."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    static = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    texts = ["word1 unknownword", "unknownword word2 otherunknown"]
    expected = static.tokenize(texts)

    assert [row[row != s.pad_id].tolist() for row in s.tokenize(texts)] == expected
    assert s._batch_tokenizer(None)(texts) == expected


def test_tokenize_without_unk_token(mock_vectors: np.ndarray) -> None:
    """A tokenizer with no unk token has nothing to drop, so tokenization is left as is."""
    vocab = ["[PAD]", "word1", "word2", "word3"]
    tokenizer = Tokenizer(BPE(vocab={token: idx for idx, token in enumerate(vocab)}, merges=[], ignore_merges=True))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
    vectors = mock_vectors[: len(vocab)]

    s = StaticModelForClassification(vectors=torch.from_numpy(vectors).float(), tokenizer=tokenizer)
    static = StaticModel(vectors=vectors, tokenizer=tokenizer)
    assert s.unk_token_id is None

    texts = ["word1 word2", "word3 word1 word2"]
    expected = static.tokenize(texts)

    assert [row[row != s.pad_id].tolist() for row in s.tokenize(texts)] == expected
    assert s._batch_tokenizer(None)(texts) == expected


def test_max_length_is_not_capped_by_the_static_model(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """A trainer's `max_length` is its own: the static model's truncation must not cap it, in training or after export."""
    static = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer, max_length=2)
    texts = ["word1 word2 word3 word1 word2 word3"]
    assert len(static.tokenize(texts)[0]) == 2

    s = StaticModelForClassification.from_static_model(model=static, max_length=4)
    assert s.tokenize(texts).shape[1] == 4
    assert [len(row) for row in s._batch_tokenizer(None)(texts)] == [6]
    assert len(s.to_static_model().tokenize(texts)[0]) == 4

    # The static model keeps its own setting, and the trainer keeps its own once the static model changes.
    assert len(static.tokenize(texts)[0]) == 2
    static.max_length = None
    assert s.tokenize(texts).shape[1] == 4


def test_predict(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the predict function."""
    result = mock_trained_pipeline.predict(["dog cat", "dog"]).tolist()
    if mock_trained_pipeline.multilabel:
        if type(mock_trained_pipeline.classes_[0]) == str:
            assert result == [["b"], ["b"]]
        else:
            assert result == [[1], [1]]
    else:
        if type(mock_trained_pipeline.classes_[0]) == str:
            assert result == ["b", "b"]
        else:
            assert result == [1, 1]


def test_predict_proba(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the predict function."""
    result = mock_trained_pipeline.predict_proba(["dog cat", "dog"])
    assert result.shape == (2, 2)


def test_convert_to_pipeline(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Convert a model to a pipeline."""
    mock_trained_pipeline.eval()
    pipeline = mock_trained_pipeline.to_pipeline()
    encoded_pipeline = pipeline.model.encode(["dog cat", "dog"])
    encoded_model = mock_trained_pipeline._encode(mock_trained_pipeline.tokenize(["dog cat", "dog"])).detach().numpy()
    assert np.allclose(encoded_pipeline, encoded_model)
    a = pipeline.predict(["dog cat", "dog"]).tolist()
    b = mock_trained_pipeline.predict(["dog cat", "dog"]).tolist()
    assert a == b
    p1 = pipeline.predict_proba(["dog cat", "dog"])
    p2 = mock_trained_pipeline.predict_proba(["dog cat", "dog"])
    assert np.allclose(p1, p2, rtol=1e-5)


def test_convert_to_pipeline_similarity(mock_trained_similarity_pipeline: StaticModelForSimilarity) -> None:
    """Convert a model to a pipeline."""
    mock_trained_similarity_pipeline.eval()
    pipeline = mock_trained_similarity_pipeline.to_pipeline()
    encoded_pipeline = pipeline.model.encode(["dog cat", "dog"])
    encoded_model = (
        mock_trained_similarity_pipeline._encode(mock_trained_similarity_pipeline.tokenize(["dog cat", "dog"]))
        .detach()
        .numpy()
    )
    assert np.allclose(encoded_pipeline, encoded_model)
    p1 = pipeline.predict(["dog cat", "dog"])
    p2 = mock_trained_similarity_pipeline.encode(["dog cat", "dog"])
    assert np.allclose(p1, p2, rtol=1e-5, atol=1e-4)


def test_convert_to_pipeline_regression(mock_trained_regression_pipeline: StaticModelForRegression) -> None:
    """Convert a model to a pipeline."""
    mock_trained_regression_pipeline.eval()
    pipeline = mock_trained_regression_pipeline.to_pipeline()
    encoded_pipeline = pipeline.model.encode(["dog cat", "dog"])
    encoded_model = (
        mock_trained_regression_pipeline._encode(mock_trained_regression_pipeline.tokenize(["dog cat", "dog"]))
        .detach()
        .numpy()
    )
    assert np.allclose(encoded_pipeline, encoded_model)
    p1 = pipeline.predict(["dog cat", "dog"])
    p2 = mock_trained_regression_pipeline.encode(["dog cat", "dog"])
    assert np.allclose(p1, p2, rtol=1e-5, atol=1e-4)


def test_pairdataset_init() -> None:
    """Test the pair dataset init."""
    dataset = PairDataset(ColumnRows(text_a=[[0], [1]], text_b=[[2], [3]]), _pretokenized)
    assert len(dataset) == 2


def test_pairdataset_init_incorrect() -> None:
    """Test the pair dataset init with mismatched lengths."""
    with pytest.raises(ValueError):
        PairDataset(ColumnRows(text_a=[[0]], text_b=[[2], [3]]), _pretokenized)


def test_pairdataset_collate() -> None:
    """Batches should stack the two padded halves into a single (2, batch, seq_len) tensor."""
    dataset = PairDataset(ColumnRows(text_a=[[1], [1, 2]], text_b=[[1, 2, 3], [1]]), _pretokenized, pad_id=0)
    batch, y = next(iter(dataset.to_dataloader(shuffle=False, batch_size=2)))
    assert batch.shape == (2, 2, 3)
    assert torch.equal(y, torch.tensor([0, 1]))
    assert torch.equal(batch[0], torch.tensor([[1, 0, 0], [1, 2, 0]]))
    assert torch.equal(batch[1], torch.tensor([[1, 2, 3], [1, 0, 0]]))


def _distinct(n: int) -> torch.Tensor:
    return torch.arange(n)


def test_pair_infonce_loss_is_query_to_document() -> None:
    """The loss is the cross-entropy of each first text over all second texts in the batch."""
    torch.manual_seed(0)
    out_a, out_b = torch.randn(4, 3), torch.randn(4, 3)
    loss_fn = PairInfoNCELoss(temperature=0.1)
    logits = torch.nn.functional.normalize(out_a, dim=1) @ torch.nn.functional.normalize(out_b, dim=1).T / 0.1
    expected = torch.nn.functional.cross_entropy(logits, torch.arange(4))

    loss = loss_fn((out_a, out_b, _distinct(4), _distinct(4)), torch.arange(4))
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)


def test_pair_infonce_loss_masks_duplicate_positives() -> None:
    """Second texts identical to an anchor's positive are not used as negatives for that anchor."""
    out_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out_b = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss_fn = PairInfoNCELoss(temperature=0.05)

    loss = loss_fn((out_a, out_b, _distinct(2), torch.tensor([0, 0])), torch.arange(2))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pair_infonce_loss_prefers_aligned_pairs() -> None:
    """InfoNCE is low when each anchor matches its own positive and high when it matches another."""
    loss_fn = PairInfoNCELoss(temperature=0.05)
    out_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    swapped = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    assert loss_fn((out_a, aligned, _distinct(2), _distinct(2)), torch.arange(2)).item() == pytest.approx(0.0, abs=1e-6)
    assert loss_fn((out_a, swapped, _distinct(2), _distinct(2)), torch.arange(2)).item() == pytest.approx(
        20.0, abs=1e-4
    )


def test_pair_infonce_loss_masks_other_positives_of_the_same_anchor() -> None:
    """Pairs with an identical first text are all positives for it, so they don't compete."""
    out_a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    out_b = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    loss_fn = PairInfoNCELoss(temperature=0.05)

    loss = loss_fn((out_a, out_b, torch.tensor([0, 0]), _distinct(2)), torch.arange(2))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pair_infonce_loss_keeps_distinct_but_parallel_negatives() -> None:
    """Distinct second texts are negatives, even if their embeddings are nearly parallel."""
    out_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out_b = torch.tensor([[1.0, 0.0], [1.0, 1e-4]])
    loss_fn = PairInfoNCELoss(temperature=0.05)

    loss = loss_fn((out_a, out_b, _distinct(2), _distinct(2)), torch.arange(2))
    assert loss.item() == pytest.approx(float(np.log(2)), abs=1e-3)


@pytest.mark.parametrize("temperature", [0.0, -0.05])
def test_pair_infonce_loss_rejects_non_positive_temperature(temperature: float) -> None:
    """The temperature must be positive."""
    with pytest.raises(ValueError):
        PairInfoNCELoss(temperature=temperature)


def test_pair_similarity_out_dim_defaults_to_embed_dim(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """The output dimension defaults to the input embedding dimension when not specified."""
    s = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    assert s.out_dim == mock_vectors.shape[1]

    s = StaticModelForPairSimilarity(
        vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer, out_dim=7
    )
    assert s.out_dim == 7


@pytest.mark.parametrize(
    "model_class", [StaticModelForSimilarity, StaticModelForRegression, StaticModelForPairSimilarity]
)
def test_no_head_without_layers(model_class: Any, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Without layers and with an unchanged dimension, the model has no head and matches its static model."""
    model = model_class(
        vectors=torch.from_numpy(mock_vectors).float(),
        tokenizer=mock_tokenizer,
        n_layers=0,
        out_dim=mock_vectors.shape[1],
    )
    assert len(model.head) == 0

    texts = ["dog cat", "dog"]
    np.testing.assert_allclose(model.encode(texts), model.to_static_model().encode(texts), atol=1e-6)
    np.testing.assert_allclose(model.encode(texts), model.to_pipeline().predict(texts), atol=1e-6)


@pytest.mark.parametrize(
    "model_class", [StaticModelForSimilarity, StaticModelForRegression, StaticModelForPairSimilarity]
)
def test_head_without_layers_changes_dimension(
    model_class: Any, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Without layers but with a different output dimension, the head is a single linear layer."""
    model = model_class(
        vectors=torch.from_numpy(mock_vectors).float(),
        tokenizer=mock_tokenizer,
        n_layers=0,
        out_dim=mock_vectors.shape[1] + 1,
    )
    assert len(model.head) == 1
    assert isinstance(model.head[0], torch.nn.Linear)


def test_similarity_fit_without_layers(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """The head of a similarity model follows the dimension of the targets it is fit on."""
    model = StaticModelForSimilarity(
        vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer, n_layers=0
    )
    texts = ["word1", "word2", "word3", "word1 word2"] * 2
    model.fit(texts, torch.randn(len(texts), mock_vectors.shape[1]), max_epochs=1)
    assert len(model.head) == 0

    model.fit(texts, torch.randn(len(texts), mock_vectors.shape[1] + 1), max_epochs=1)
    assert isinstance(model.head[0], torch.nn.Linear)


def test_classifier_keeps_head_when_dimensions_match(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """A classifier without layers keeps its linear layer, even if the number of classes equals the dimension."""
    model = StaticModelForClassification(
        vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer, n_layers=0
    )
    assert model.out_dim == mock_vectors.shape[1]
    assert isinstance(model.head[0], torch.nn.Linear)


def test_pair_similarity_forward(mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity) -> None:
    """The forward pass should return one head output per half of the pair batch."""
    model = mock_trained_pair_similarity_pipeline
    dataset = model._pair_dataset(ColumnRows(text_a=["dog cat", "dog"], text_b=["puppy", "kitten cat"]))
    batch, _ = next(iter(dataset.to_dataloader(shuffle=False, batch_size=2)))

    with torch.no_grad():
        out_a, out_b, ids_a, ids_b = model(batch)
    assert out_a.shape == (2, model.out_dim)
    assert out_b.shape == (2, model.out_dim)
    assert ids_a.tolist()[0] != ids_a.tolist()[1]
    assert ids_b.tolist()[0] != ids_b.tolist()[1]


def test_pair_similarity_mismatched_lengths(
    mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity,
) -> None:
    """text_a and text_b must have the same length."""
    with pytest.raises(ValueError):
        mock_trained_pair_similarity_pipeline.fit(["dog", "cat"], ["puppy"])


def test_pair_similarity_val_split_errors(mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity) -> None:
    """Both validation halves must be provided together, or neither."""
    with pytest.raises(ValueError):
        mock_trained_pair_similarity_pipeline.fit(
            ["dog", "cat"], ["puppy", "kitten"], text_a_val=["dog"], text_b_val=None
        )
    with pytest.raises(ValueError):
        mock_trained_pair_similarity_pipeline.fit(
            ["dog", "cat"], ["puppy", "kitten"], text_a_val=["dog", "cat"], text_b_val=["puppy"]
        )


def test_pair_similarity_fit_with_explicit_val(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """A model can be fit with explicit validation pairs instead of an automatic split."""
    model = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    text_a = ["word1", "word2", "word3", "word1 word2"]
    text_b = ["word2", "word3", "word1", "word3 word1"]
    model.fit(
        text_a,
        text_b,
        text_a_val=["word1", "word3"],
        text_b_val=["word2", "word1"],
        early_stopping_patience=1,
        max_epochs=1,
    )


def test_pair_similarity_fit_rejects_splits_it_cannot_learn_from(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Both splits need at least two pairs."""
    model = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="training set needs at least two pairs"):
        model.fit(["word1", "word2"], ["word2", "word3"])
    with pytest.raises(ValueError, match="validation set needs at least two pairs"):
        model.fit(["word1", "word2"], ["word2", "word3"], text_a_val=["word1"], text_b_val=["word3"])


def test_pair_similarity_fit_rejects_invalid_temperature_and_batch_size(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """A non-positive temperature and a batch size of 1 are rejected."""
    model = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    text_a, text_b = ["word1", "word2", "word3", "word1 word2"], ["word2", "word3", "word1", "word3"]
    with pytest.raises(ValueError, match="temperature"):
        model.fit(text_a, text_b, test_size=0.5, temperature=0.0)
    with pytest.raises(ValueError, match="batch_size"):
        model.fit(text_a, text_b, test_size=0.5, batch_size=1)


def test_pairdataset_drops_single_pair_batches() -> None:
    """A final batch with a single pair is dropped, unless it is the only pair."""
    dataset = PairDataset(ColumnRows(text_a=[[1], [2], [3]], text_b=[[1], [2], [3]]), _pretokenized)
    assert [len(y) for _, y in dataset.to_dataloader(shuffle=False, batch_size=2)] == [2]
    single = PairDataset(ColumnRows(text_a=[[1]], text_b=[[1]]), _pretokenized)
    assert [len(y) for _, y in single.to_dataloader(shuffle=False, batch_size=2)] == [1]


def test_pair_similarity_forward_ids_identical_texts(
    mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity,
) -> None:
    """Identical texts in a batch get the same id."""
    model = mock_trained_pair_similarity_pipeline
    dataset = model._pair_dataset(ColumnRows(text_a=["dog", "cat", "dog"], text_b=["puppy", "puppy", "kitten"]))
    batch, _ = next(iter(dataset.to_dataloader(shuffle=False, batch_size=3)))
    with torch.no_grad():
        _, _, ids_a, ids_b = model(batch)
    assert ids_a[0] == ids_a[2] and ids_a[0] != ids_a[1]
    assert ids_b[0] == ids_b[1] and ids_b[0] != ids_b[2]


def test_convert_to_pipeline_pair_similarity(
    mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity,
) -> None:
    """Convert a model to a pipeline."""
    mock_trained_pair_similarity_pipeline.eval()
    pipeline = mock_trained_pair_similarity_pipeline.to_pipeline()
    encoded_pipeline = pipeline.model.encode(["dog cat", "dog"])
    encoded_model = (
        mock_trained_pair_similarity_pipeline._encode(
            mock_trained_pair_similarity_pipeline.tokenize(["dog cat", "dog"])
        )
        .detach()
        .numpy()
    )
    assert np.allclose(encoded_pipeline, encoded_model)
    p1 = pipeline.predict(["dog cat", "dog"])
    p2 = mock_trained_pair_similarity_pipeline.encode(["dog cat", "dog"])
    assert np.allclose(p1, p2, rtol=1e-5, atol=1e-4)


def test_y_val_none() -> None:
    """Test the y_val function."""
    tokenizer = AutoTokenizer.from_pretrained("tests/data/test_tokenizer").backend_tokenizer
    torch.random.manual_seed(42)
    vectors_torched = torch.randn(len(tokenizer.get_vocab()), 12)
    model = StaticModelForClassification(vectors=vectors_torched, tokenizer=tokenizer, hidden_dim=12).to("cpu")

    X = ["dog", "cat"]
    y = ["0", "1"]

    X_val = ["dog", "cat"]
    y_val = ["0", "1"]

    with pytest.raises(ValueError):
        model.fit(X, y, X_val=X_val, y_val=None)
    with pytest.raises(ValueError):
        model.fit(X, y, X_val=None, y_val=y_val)
    model.fit(X, y, X_val=None, y_val=None)


def test_class_weight() -> None:
    """Test the class weight function."""
    tokenizer = AutoTokenizer.from_pretrained("tests/data/test_tokenizer").backend_tokenizer
    torch.random.manual_seed(42)
    vectors_torched = torch.randn(len(tokenizer.get_vocab()), 12)
    model = StaticModelForClassification(vectors=vectors_torched, tokenizer=tokenizer, hidden_dim=12).to("cpu")

    X = ["dog", "cat"]
    y = ["0", "1"]

    bad_class_weight = torch.tensor([1.0])
    with pytest.raises(ValueError):
        model.fit(X, y, class_weight=bad_class_weight)

    class_weight = torch.tensor([1.0, 2.0])
    model.fit(X, y, class_weight=class_weight)


@pytest.mark.parametrize(
    "y_multi,y_val_multi,should_crash",
    [[True, True, False], [False, False, False], [True, False, True], [False, True, True]],
)
def test_y_val(y_multi: bool, y_val_multi: bool, should_crash: bool) -> None:
    """Test the y_val function."""
    tokenizer = AutoTokenizer.from_pretrained("tests/data/test_tokenizer").backend_tokenizer
    torch.random.manual_seed(42)
    vectors_torched = torch.randn(len(tokenizer.get_vocab()), 12)
    model = StaticModelForClassification(vectors=vectors_torched, tokenizer=tokenizer, hidden_dim=12).to("cpu")

    X = ["dog", "cat"]
    y = [["0", "1"], ["0"]] if y_multi else ["0", "1"]  # type: ignore

    X_val = ["dog", "cat"]
    y_val = [["0", "1"], ["0"]] if y_val_multi else ["0", "1"]  # type: ignore

    if should_crash:
        with pytest.raises(ValueError):
            model.fit(X, y, X_val=X_val, y_val=y_val)
    else:
        model.fit(X, y, X_val=X_val, y_val=y_val)


def test_evaluate(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the evaluate function."""
    if mock_trained_pipeline.multilabel:
        if type(mock_trained_pipeline.classes_[0]) == str:
            mock_trained_pipeline.evaluate(["dog cat", "dog"], [["a", "b"], ["a"]])
        else:
            # Ignore the type error since we don't support int labels in our typing, but the code does
            mock_trained_pipeline.evaluate(["dog cat", "dog"], [[0, 1], [0]])  # type: ignore
    else:
        if type(mock_trained_pipeline.classes_[0]) == str:
            mock_trained_pipeline.evaluate(["dog cat", "dog"], ["a", "a"])
        else:
            # Ignore the type error since we don't support int labels in our typing, but the code does
            mock_trained_pipeline.evaluate(["dog cat", "dog"], [1, 1])  # type: ignore


def test_get_probable_pad_token_id(mock_tokenizer: Tokenizer, caplog: pytest.LogCaptureFixture) -> None:
    """Test loading from a static model with a pad token."""
    tokenizer_model = TokenizerModel.from_tokenizer(mock_tokenizer)
    t = tokenizer_model.to_tokenizer()
    token_id = get_probable_pad_token_id(t)
    assert token_id == 0

    # Adds new token
    tokenizer_model.pad_token = "haha"
    t = tokenizer_model.to_tokenizer()
    token_id = get_probable_pad_token_id(t)
    assert token_id == 5

    tokenizer_model.pad_token = "word1"
    t = tokenizer_model.to_tokenizer()
    token_id = get_probable_pad_token_id(t)
    assert token_id == 1

    # Remove padding token
    tokenizer_model.pad_token = None
    t = tokenizer_model.to_tokenizer()
    token_id = get_probable_pad_token_id(t)
    assert token_id == tokenizer_model.vocabulary["[PAD]"]

    tokenizer_model = tokenizer_model.remove_token_from_vocabulary("[PAD]")
    t = tokenizer_model.to_tokenizer()
    with caplog.at_level(logging.WARNING, logger="model2vec.train.utils"):
        token_id = get_probable_pad_token_id(t)
    assert token_id == 0
    assert "No known pad token found, using 0 as default" in caplog.text


def test_get_probable_pad_token_id_through_static_model(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test that a non-standard pad token survives StaticModel construction."""
    tokenizer_model = TokenizerModel.from_tokenizer(mock_tokenizer)
    tokenizer_model.pad_token = "word1"
    pad_id = tokenizer_model.pad_token_id
    assert pad_id != 0

    t = tokenizer_model.to_tokenizer()
    model = StaticModel(vectors=mock_vectors, tokenizer=t)

    assert model.tokenizer.padding is not None
    assert get_probable_pad_token_id(model.tokenizer) == pad_id


def test_resolve_class_weight(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test what the class weights are."""
    w_dict = dict(zip(mock_trained_pipeline.classes, [0.5, 3]))
    c1, c2 = mock_trained_pipeline.classes_
    counts = Counter({c1: 100, c2: 50})
    w = mock_trained_pipeline._resolve_class_weight(w_dict, counts)
    assert isinstance(w, torch.Tensor)
    assert w.tolist() == [0.5, 3]

    w = mock_trained_pipeline._resolve_class_weight("balanced", counts)
    assert isinstance(w, torch.Tensor)
    assert w.tolist() == [0.75, 1.5]

    assert mock_trained_pipeline._resolve_class_weight(None, counts) is None


def test_determine_interval() -> None:
    """Test the training interval and check_val_every_epoch are determined correctly."""
    # Lower than 250 batches, so we only check at the end of the epoch
    val_check_interval, check_val_every_epoch = StaticModelForClassification._determine_val_check_interval(
        validation_steps=None, train_length=1000, batch_size=20
    )
    assert val_check_interval is None
    assert check_val_every_epoch == 1

    # More than 250 batches, but low train batches, so we check four times per epoch
    val_check_interval, check_val_every_epoch = StaticModelForClassification._determine_val_check_interval(
        validation_steps=None, train_length=1000, batch_size=1
    )
    assert val_check_interval == 250
    assert check_val_every_epoch is None

    # More than 250 batches, but low train batches, so we check four times per epoch
    val_check_interval, check_val_every_epoch = StaticModelForClassification._determine_val_check_interval(
        validation_steps=None, train_length=1200, batch_size=1
    )
    assert val_check_interval == 300
    assert check_val_every_epoch is None

    val_check_interval, check_val_every_epoch = StaticModelForClassification._determine_val_check_interval(
        validation_steps=None, train_length=100000, batch_size=20
    )
    assert val_check_interval == 1250
    assert check_val_every_epoch is None

    # Set by the user, so nothing matters.
    val_check_interval, check_val_every_epoch = StaticModelForClassification._determine_val_check_interval(
        validation_steps=100, train_length=1000, batch_size=32
    )
    assert val_check_interval == 100
    assert check_val_every_epoch is None


def test_logit() -> None:
    """Test on random data."""
    x = torch.arange(10).float() / 10
    assert torch.allclose(logit(torch.sigmoid(x)), x, atol=1e-6)


def test_seed_everything_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """seed_everything also seeds CUDA RNGs when CUDA is available."""
    seeded_with: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded_with.append)
    seed_everything(123)
    assert seeded_with and all(seed == 123 for seed in seeded_with)


def test_resolve_max_epochs() -> None:
    """None and negative max_epochs resolve to a large cap; positive values pass through unchanged."""
    assert _resolve_max_epochs(None) > 0
    assert _resolve_max_epochs(-1) > 0
    assert _resolve_max_epochs(3) == 3


def test_resolve_device_explicit() -> None:
    """An explicit device string is returned unchanged."""
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """'auto' picks cuda when available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """'auto' falls back to cpu when no accelerator is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def _make_loader(n: int, in_dim: int = 3, out_dim: int = 2) -> DataLoader:
    x = torch.randn(n, in_dim)
    y = torch.randn(n, out_dim)
    return DataLoader(TensorDataset(x, y), batch_size=1)


def test_run_training_loop_without_early_stopping_stops_at_max_epochs() -> None:
    """With early stopping disabled, training runs until max_epochs and stops there."""
    model = nn.Linear(3, 2)
    state_dict = run_training_loop(
        model=model,
        loss_function=nn.MSELoss(),
        learning_rate=1e-3,
        val_metric="val_loss",
        early_stopping_direction="min",
        train_loader=_make_loader(4),
        val_loader=_make_loader(2),
        early_stopping_patience=None,
        min_epochs=None,
        max_epochs=1,
        device=resolve_device("cpu"),
        val_check_interval=None,
        check_val_every_epoch=1,
    )
    assert set(state_dict) == set(model.state_dict())


def test_run_training_loop_mid_epoch_early_stop() -> None:
    """Early stopping can trigger mid-epoch via val_check_interval, not just at epoch boundaries."""
    model = nn.Linear(3, 2)
    state_dict = run_training_loop(
        model=model,
        loss_function=nn.MSELoss(),
        learning_rate=1e-3,
        val_metric="val_loss",
        early_stopping_direction="min",
        train_loader=_make_loader(4),
        val_loader=_make_loader(2),
        early_stopping_patience=1,
        min_epochs=None,
        max_epochs=None,
        device=resolve_device("cpu"),
        val_check_interval=1,
        check_val_every_epoch=None,
        compute_metrics=lambda head_out, y, loss: {"loss": 1.0},
    )
    assert set(state_dict) == set(model.state_dict())


def test_run_training_loop_steps_scheduler_once_per_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LR scheduler steps once per epoch, not once per validation check."""
    step_calls: list[float] = []
    original_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def counting_step(self: torch.optim.lr_scheduler.ReduceLROnPlateau, metrics: float) -> None:
        step_calls.append(metrics)
        original_step(self, metrics)

    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", counting_step)

    model = nn.Linear(3, 2)
    run_training_loop(
        model=model,
        loss_function=nn.MSELoss(),
        learning_rate=1e-3,
        val_metric="val_loss",
        early_stopping_direction="min",
        train_loader=_make_loader(6),
        val_loader=_make_loader(2),
        early_stopping_patience=None,
        min_epochs=None,
        max_epochs=3,
        device=resolve_device("cpu"),
        val_check_interval=1,
        check_val_every_epoch=None,
    )
    assert len(step_calls) == 3


def test_split_indices() -> None:
    """The split is disjoint, sorted, complete, and has the requested test size."""
    train, test = split_indices(10, 0.3)
    assert len(test) == 3
    assert sorted([*train, *test]) == list(range(10))
    assert list(train) == sorted(train)
    assert list(test) == sorted(test)


def test_split_indices_absolute_and_capped_sizes() -> None:
    """An int test size is a number of items, and a fractional test size can be capped."""
    assert len(split_indices(100, 7)[1]) == 7
    assert len(split_indices(100, 0.5, max_test_size=10)[1]) == 10
    assert len(split_indices(100, 0.05, max_test_size=10)[1]) == 5
    assert len(split_indices(100, 30, max_test_size=10)[1]) == 30


def test_split_indices_stratified() -> None:
    """A list of single labels is split per label, unless a label occurs only once."""
    labels = ["a"] * 6 + ["b"] * 4
    train, test = split_indices(len(labels), 0.5, stratify_by=labels)
    assert sorted(labels[i] for i in test) == ["a"] * 3 + ["b"] * 2
    assert sorted([*train, *test]) == list(range(10))

    labels = ["a"] * 9 + ["b"]
    assert len(split_indices(len(labels), 0.5, stratify_by=labels)[1]) == 5


def test_split_indices_numpy_int_and_bool() -> None:
    """A numpy int test size is a number of items, and a bool test size is rejected."""
    assert len(split_indices(100, np.int64(7), max_test_size=50)[1]) == 7
    with pytest.raises(ValueError):
        split_indices(100, True)


def test_split_indices_stratified_respects_test_size() -> None:
    """A stratified split holds out exactly the requested number of items, even if it is capped."""
    labels = [str(i % 20) for i in range(1000)] + ["rare"] * 2
    train, test = split_indices(len(labels), 0.5, max_test_size=30, stratify_by=labels)
    assert len(test) == 30
    assert {labels[i] for i in test} == set(labels)
    assert sorted([*train, *test]) == list(range(len(labels)))

    labels = [str(i % 20) for i in range(1000)]
    assert len(split_indices(len(labels), 10, stratify_by=labels)[1]) == 10


@pytest.mark.parametrize("labels", [["a"] * 30 + ["b"] * 10, [0] * 30 + [1] * 10])
def test_split_indices_stratifies_columns_like_lists(labels: list[Any]) -> None:
    """Columns of single labels are stratified exactly like lists, also after a selection or when nested."""
    dataset = Dataset.from_dict({"label": labels, "meta": [{"label": label} for label in labels]})
    selected = dataset.shuffle(seed=0).select(range(30))
    for column in (dataset["label"], dataset["meta"]["label"], selected["label"]):
        expected = split_indices(len(column), 0.25, stratify_by=list(column))
        actual = split_indices(len(column), 0.25, stratify_by=column)
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    test = split_indices(len(labels), 0.5, stratify_by=dataset["label"])[1]
    assert sorted(dataset["label"][test.tolist()]) == labels[:1] * 15 + labels[-1:] * 5


@pytest.mark.parametrize("labels", [["b"] * 30 + ["a"] * 10, [1] * 30 + [0] * 10])
def test_split_indices_stratifies_arrays_like_lists(labels: list[Any]) -> None:
    """Arrays and tensors of single labels are stratified exactly like lists."""
    expected = split_indices(len(labels), 0.25, stratify_by=labels)
    arrays: list[Any] = [np.array(labels)]
    if isinstance(labels[0], int):
        arrays.append(torch.tensor(labels))
    for array in arrays:
        actual = split_indices(len(array), 0.25, stratify_by=array)
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize(
    "labels", [[["a"], ["b"]] * 10, ["a"] * 19 + ["b"], [[0.5, 1.0]] * 20], ids=["multilabel", "singleton", "vectors"]
)
def test_split_indices_does_not_stratify_other_columns(labels: list[Any]) -> None:
    """Columns of multi-labels or vectors, or with a class that occurs once, are split at random."""
    column = Dataset.from_dict({"label": labels})["label"]
    expected = split_indices(len(labels), 0.25)
    assert all(np.array_equal(a, b) for a, b in zip(split_indices(len(labels), 0.25, stratify_by=column), expected))


@pytest.mark.parametrize("labels", [[["a"], ["b"]] * 10, [[0.5, 1.0]] * 20], ids=["multilabel", "vectors"])
def test_split_indices_does_not_read_unstratifiable_columns(labels: list[Any]) -> None:
    """Columns that can't be stratified are not read."""
    column = Dataset.from_dict({"label": labels})["label"]
    with patch("model2vec.train.utils.read_column") as read_column_mock:
        split_indices(len(labels), 0.25, stratify_by=column)
    read_column_mock.assert_not_called()


def test_column_rows_reads_other_sequences() -> None:
    """Generic sequences are read item by item."""
    rows = ColumnRows(text=UserList(["a", "b", "c"]), label=np.array([0, 1, 2]))
    fetched = rows[[2, 0]]
    assert fetched["text"] == ["c", "a"]
    assert fetched["label"].tolist() == [2, 0]


def test_column_rows_reads_hf_columns() -> None:
    """Rows can be read from the columns of a Hugging Face dataset."""
    dataset = Dataset.from_dict({"a": ["x", "y", "z"], "b": [[1.0], [2.0], [3.0]]})
    rows = ColumnRows(text=dataset["a"], label=dataset["b"])
    assert len(rows) == 3
    assert rows[[2, 0]] == {"text": ["z", "x"], "label": [[3.0], [1.0]]}


def _assert_same_weights(expected: nn.Module, actual: nn.Module) -> None:
    for (name, expected_tensor), actual_tensor in zip(expected.state_dict().items(), actual.state_dict().values()):
        assert torch.equal(expected_tensor, actual_tensor), name


_TRAIN_TEXTS = ["word1", "word2", "word3", "word1 word2", "word2 word3", "word3 word1", "word1 word3", "word2"]
_VAL_TEXTS = ["word1 word2", "word3"]


def test_pair_similarity_fit_on_columns_matches_lists(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Training on the columns of a dataset gives the same model as training on the same pairs as lists."""
    text_a = ["word1", "word2", "word3", "word1 word2", "word2 word3", "word3 word1"]
    text_b = ["word2", "word3", "word1", "word3 word1", "word1", "word2 word2"]
    val_a, val_b = ["word1", "word3"], ["word2", "word1"]

    from_lists = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    from_lists.fit(text_a, text_b, text_a_val=val_a, text_b_val=val_b, max_epochs=3, batch_size=2, device="cpu")

    dataset = Dataset.from_dict({"a": text_a, "b": text_b})
    val_dataset = Dataset.from_dict({"a": val_a, "b": val_b})
    from_columns = StaticModelForPairSimilarity(
        vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer
    )
    from_columns.fit(
        dataset["a"],
        dataset["b"],
        text_a_val=val_dataset["a"],
        text_b_val=val_dataset["b"],
        max_epochs=3,
        batch_size=2,
        device="cpu",
    )

    _assert_same_weights(from_lists, from_columns)


def test_pair_similarity_fit_on_columns_with_split(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Without validation pairs, the validation pairs are split off from the columns."""
    model = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    dataset = Dataset.from_dict({"a": ["word1", "word2", "word3", "word1 word2"], "b": ["word2", "word3", "word1", ""]})
    model.fit(dataset["a"], dataset["b"], test_size=2, max_epochs=1, device="cpu")


@pytest.mark.parametrize(
    "labels, val_labels",
    [
        (["a", "b", "a", "c", "b", "a", "c", "a"], ["a", "b"]),
        ([["a"], ["b", "c"], ["a", "b"], [], ["c"], ["a"], ["b"], ["a", "c"]], [["a"], ["b"]]),
    ],
)
def test_classifier_fit_on_columns_matches_lists(
    mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, labels: list, val_labels: list
) -> None:
    """Training a classifier on the columns of a dataset gives the same model as training on lists."""
    from_lists = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    from_lists.fit(
        _TRAIN_TEXTS, labels, X_val=_VAL_TEXTS, y_val=val_labels, class_weight="balanced", max_epochs=3, batch_size=2
    )

    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": labels})
    val_dataset = Dataset.from_dict({"text": _VAL_TEXTS, "label": val_labels})
    from_columns = StaticModelForClassification(
        vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer
    )
    from_columns.fit(
        dataset["text"],
        dataset["label"],
        X_val=val_dataset["text"],
        y_val=val_dataset["label"],
        class_weight="balanced",
        max_epochs=3,
        batch_size=2,
    )

    assert from_columns.classes_ == from_lists.classes_
    assert from_columns.multilabel == from_lists.multilabel
    _assert_same_weights(from_lists, from_columns)


def test_regressor_fit_on_columns_matches_lists(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Training a regressor on the columns of a dataset gives the same model as training on tensors."""
    y = torch.randn(len(_TRAIN_TEXTS), 3)
    y_val = torch.randn(len(_VAL_TEXTS), 3)

    from_tensors = StaticModelForRegression(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    from_tensors.fit(_TRAIN_TEXTS, y, X_val=_VAL_TEXTS, y_val=y_val, max_epochs=3, batch_size=2)

    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": y.tolist()})
    val_dataset = Dataset.from_dict({"text": _VAL_TEXTS, "label": y_val.tolist()})
    from_columns = StaticModelForRegression(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    from_columns.fit(
        dataset["text"],
        dataset["label"],
        X_val=val_dataset["text"],
        y_val=val_dataset["label"],
        max_epochs=3,
        batch_size=2,
    )

    assert from_columns.out_dim == 3
    _assert_same_weights(from_tensors, from_columns)


def test_fit_on_columns_with_split(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Without validation data, the validation data is split off from the columns."""
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": [0, 1] * 4, "vector": [[0.5, 1.0]] * 8})
    classifier = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    classifier.fit(dataset["text"], dataset["label"], max_epochs=1)
    assert classifier.classes_ == [0, 1]

    regressor = StaticModelForRegression(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    regressor.fit(dataset["text"], dataset["vector"], max_epochs=1)
    assert regressor.out_dim == 2


def test_classifier_rejects_float_labels_in_column(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Labels in a column must be strings, integers, or lists of those."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": [0.5] * 8})
    with pytest.raises(ValueError):
        model.fit(dataset["text"], dataset["label"])


@pytest.mark.parametrize("model_class", [StaticModelForClassification, StaticModelForRegression])
def test_fit_rejects_datasets(model_class: type, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """`fit` takes columns, not whole datasets."""
    model = model_class(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    labels: list = [[0.5, 1.0]] * 8 if model_class is StaticModelForRegression else ["a", "b"] * 4
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": labels})
    with pytest.raises(ValueError, match="columns"):
        model.fit(dataset, labels)
    with pytest.raises(ValueError, match="columns"):
        model.fit(_TRAIN_TEXTS, dataset)
    with pytest.raises(ValueError, match="columns"):
        model.fit(_TRAIN_TEXTS, labels, X_val=DatasetDict({"train": dataset}), y_val=labels)


def test_pair_similarity_fit_rejects_datasets(
    mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity,
) -> None:
    """`fit` takes columns, not whole datasets."""
    model = mock_trained_pair_similarity_pipeline
    dataset = Dataset.from_dict({"a": ["word1", "word2"], "b": ["word2", "word3"]})
    with pytest.raises(ValueError, match="columns"):
        model.fit(dataset, ["word1", "word2"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="columns"):
        model.fit(["word1", "word2"], ["word2", "word3"], text_a_val=dataset, text_b_val=["x"])  # type: ignore[arg-type]


def test_fit_caps_validation_split(
    monkeypatch: pytest.MonkeyPatch, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """With a fractional test size, fit holds out at most MAX_VALIDATION_SIZE rows."""
    monkeypatch.setattr("model2vec.train.base.MAX_VALIDATION_SIZE", 2)
    monkeypatch.setattr("model2vec.train.pairs.MAX_VALIDATION_SIZE", 2)
    sizes: list[tuple[int, int]] = []

    def fake_run_training_loop(**kwargs: Any) -> dict[str, torch.Tensor]:
        sizes.append((len(kwargs["train_loader"].dataset), len(kwargs["val_loader"].dataset)))
        return kwargs["model"].state_dict()

    monkeypatch.setattr("model2vec.train.base.run_training_loop", fake_run_training_loop)
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": ["a", "b"] * 4})
    classifier = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    classifier.fit(_TRAIN_TEXTS, ["a", "b"] * 4, test_size=0.5)
    classifier.fit(dataset["text"], dataset["label"], test_size=0.5)
    classifier.fit(dataset["text"], dataset["label"], test_size=3)
    pairs = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    pairs.fit(_TRAIN_TEXTS, _TRAIN_TEXTS, test_size=0.5)
    assert sizes == [(6, 2), (6, 2), (5, 3), (6, 2)]


def test_classifier_checks_validation_labels(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Validation labels that don't match the training labels are rejected before training."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="not in y"):
        model.fit(_TRAIN_TEXTS, ["a", "b"] * 4, X_val=_VAL_TEXTS, y_val=["a", "c"])
    with pytest.raises(ValueError, match="multi-label"):
        model.fit(_TRAIN_TEXTS, ["a", "b"] * 4, X_val=_VAL_TEXTS, y_val=[["a"], ["b"]])
    dataset = Dataset.from_dict({"text": _VAL_TEXTS, "label": ["a", "c"]})
    with pytest.raises(ValueError, match="not in y"):
        model.fit(_TRAIN_TEXTS, ["a", "b"] * 4, X_val=dataset["text"], y_val=dataset["label"])


@pytest.mark.parametrize("labels", [["a", None] * 4, [["a"], None] * 4, [["a"], ["b", None]] * 4])
def test_classifier_rejects_missing_labels(
    labels: list[Any], mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Missing labels are rejected before training, in lists and in columns."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": labels})
    with pytest.raises(ValueError):
        model.fit(dataset["text"], dataset["label"])
    with pytest.raises(ValueError):
        model.fit(_TRAIN_TEXTS, labels)


def test_fit_rejects_missing_texts(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Missing texts are rejected before training, in lists and in columns."""
    texts = [*_TRAIN_TEXTS[:-1], None]
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    dataset = Dataset.from_dict({"text": texts, "label": ["a", "b"] * 4})
    with pytest.raises(ValueError, match="X must be strings"):
        model.fit(dataset["text"], dataset["label"])
    with pytest.raises(ValueError, match="X_val must be strings"):
        model.fit(_TRAIN_TEXTS, ["a", "b"] * 4, X_val=["word1", None], y_val=["a", "b"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="X must be strings"):
        model.fit(dataset.shuffle(seed=0)["text"], dataset.shuffle(seed=0)["label"])
    filtered = dataset.filter(lambda row: row["text"] is not None)
    model.fit(filtered["text"], filtered["label"], max_epochs=1)
    pairs = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="text_b must be strings"):
        pairs.fit(_TRAIN_TEXTS, texts)  # type: ignore[arg-type]


def test_fit_names_mismatched_lengths(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Columns of different lengths are reported by their argument names."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="X and y must have the same length"):
        model.fit(_TRAIN_TEXTS, ["a", "b"] * 3)
    with pytest.raises(ValueError, match="X_val and y_val must have the same length"):
        model.fit(_TRAIN_TEXTS, ["a", "b"] * 4, X_val=_VAL_TEXTS, y_val=["a"])
    pairs = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="text_a_val and text_b_val must have the same length"):
        pairs.fit(_TRAIN_TEXTS, _TRAIN_TEXTS, text_a_val=_VAL_TEXTS, text_b_val=["word1"])


def test_classifier_fit_on_nested_columns(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Labels can be read from nested columns."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "meta": [{"label": "a"}, {"label": "b"}] * 4})
    model.fit(dataset["text"], dataset["meta"]["label"], max_epochs=1)
    assert model.classes_ == ["a", "b"]


@pytest.mark.parametrize("data_format", [None, "torch", "numpy"])
def test_classifier_fit_on_formatted_columns(
    data_format: str | None, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Columns of a dataset with a torch or numpy format can be used for training and validation."""
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": [0, 1] * 4}).with_format(data_format)
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model.fit(dataset["text"], dataset["label"], X_val=_VAL_TEXTS, y_val=[0, 1], max_epochs=1)  # type: ignore[arg-type]
    assert model.classes_ == [0, 1]


@pytest.mark.parametrize("data_format", [None, "torch", "numpy"])
def test_classifier_fit_on_formatted_multilabel_columns(
    data_format: str | None, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Multi-label columns of a dataset with a torch or numpy format can be used for training."""
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": [[1, 2], [1]] * 4}).with_format(data_format)
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model.fit(dataset["text"], dataset["label"], max_epochs=1)  # type: ignore[arg-type]
    assert model.multilabel
    assert model.classes_ == [1, 2]


def test_classifier_fit_on_fixed_size_list_column(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Multi-label columns with a fixed number of labels per row are multi-label."""
    features = Features({"text": Value("string"), "label": Sequence(Value("string"), length=1)})
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": [["a"], ["b"]] * 4}, features=features)
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model.fit(dataset["text"], dataset["label"], max_epochs=1)
    assert model.multilabel
    assert model.classes_ == ["a", "b"]


@pytest.mark.parametrize(
    ("y", "y_val", "message"),
    [
        ([[0.5, 1.0]] * 7 + [[0.5]], None, "same dimension"),
        ([[0.5, 1.0]] * 7 + [None], None, "sequences of numbers"),
        ([[0.5, 1.0]] * 8, [[0.5, 1.0, 1.5]] * 2, "y_val have dimension 3"),
        (torch.ones(8), None, "2-dimensional"),
        ([], None, "must not be empty"),
    ],
)
def test_regressor_checks_vectors(
    y: Any, y_val: Any, message: str, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Vectors that are missing or have different dimensions are rejected before training."""
    model = StaticModelForRegression(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    X_val = None if y_val is None else _VAL_TEXTS
    with pytest.raises(ValueError, match=message):
        model.fit(_TRAIN_TEXTS, y, X_val=X_val, y_val=y_val)


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.5, 1.0]] * 7 + [[0.5]], "same dimension"),
        ([[0.5, 1.0]] * 7 + [None], "must not be missing"),
        ([[0.5, None]] + [[0.5, 1.0]] * 7, "must not be missing"),
        ([["a", "b"]] * 8, "lists of numbers"),
        ([0.5] * 8, "lists of numbers"),
    ],
)
def test_regressor_checks_vector_columns(
    vectors: list, message: str, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """Vectors in a column that are missing, not numbers, or have different dimensions are rejected before training."""
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "vector": vectors})
    model = StaticModelForRegression(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match=message):
        model.fit(dataset["text"], dataset["vector"])


def test_regressor_fit_on_fixed_size_vector_column(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Vectors in a column with a fixed dimension can be used for training."""
    features = Features({"text": Value("string"), "vector": Sequence(Value("float32"), length=2)})
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "vector": [[0.5, 1.0]] * 8}, features=features)
    model = StaticModelForRegression(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model.fit(dataset["text"], dataset["vector"], max_epochs=1)
    assert model.out_dim == 2


def test_fit_rejects_iterable_datasets(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Iterable datasets and their columns have no length, and are rejected."""
    iterable = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": ["a", "b"] * 4}).to_iterable_dataset()
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="X comes from an iterable"):
        model.fit(iterable["text"], ["a", "b"] * 4)
    with pytest.raises(ValueError, match="y comes from an iterable"):
        model.fit(_TRAIN_TEXTS, iterable)


def test_fit_rejects_non_sequences(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Objects that are not sequences, arrays, or tensors, such as columns of an Arrow-formatted dataset, are rejected."""
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": ["a", "b"] * 4}).with_format("arrow")
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="y must be a list, a tuple, an array, a tensor, or a column"):
        model.fit(_TRAIN_TEXTS, dataset["label"])
    with pytest.raises(ValueError, match="X must be a list"):
        model.fit(dataset["text"], ["a", "b"] * 4)
    with pytest.raises(ValueError, match="X must be a list.*got str"):
        model.fit("word1 word2", ["a", "b"] * 5)  # type: ignore[arg-type]


def test_classifier_accepts_tuple_and_list_labels(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Multi-label training and validation labels can be tuples and lists."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model.fit(_TRAIN_TEXTS, [("a",), ("b",)] * 4, X_val=_VAL_TEXTS, y_val=[["a"], ["b"]], max_epochs=1)  # type: ignore[arg-type]
    assert model.multilabel


@pytest.mark.parametrize("y", [np.array(["a", "b"] * 4), np.array([0, 1] * 4), torch.tensor([0, 1] * 4)])
def test_classifier_accepts_array_labels(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer, y: Any) -> None:
    """Single-label training and validation labels can be arrays and tensors."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model.fit(_TRAIN_TEXTS, y, X_val=_VAL_TEXTS, y_val=y[:2], max_epochs=1)
    assert not model.multilabel
    assert model.classes_ == sorted(y.tolist()[:2])


def test_read_column_follows_selection() -> None:
    """A column is read in the order of the selected rows, also when nested."""
    dataset = Dataset.from_dict({"label": list(range(10)), "meta": [{"label": i} for i in range(10)]})
    selected = dataset.shuffle(seed=0).select(range(5))
    assert read_column(selected["label"]).to_pylist() == list(selected["label"])
    assert read_column(selected["meta"]["label"]).to_pylist() == list(selected["label"])


def test_fit_rejects_transformed_columns(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Columns of a dataset with a transform are rejected, also when nested or after a selection."""
    dataset = Dataset.from_dict(
        {"text": _TRAIN_TEXTS, "label": ["a", "b"] * 4, "meta": [{"label": "a"}, {"label": "b"}] * 4}
    )
    transformed = dataset.with_transform(lambda batch: batch)
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="y is a column of a Hugging Face dataset with a transform"):
        model.fit(dataset["text"], transformed["label"])
    with pytest.raises(ValueError, match="y is a column"):
        model.fit(dataset["text"], transformed["meta"]["label"])
    with pytest.raises(ValueError, match="X is a column"):
        model.fit(transformed.shuffle(seed=0)["text"], dataset["label"])
    pairs = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    with pytest.raises(ValueError, match="text_a_val is a column"):
        pairs.fit(_TRAIN_TEXTS, _TRAIN_TEXTS, text_a_val=transformed["text"], text_b_val=dataset["text"])
    model.fit(dataset["text"], transformed.with_format(None)["label"], max_epochs=1)


def test_class_targets() -> None:
    """Class labels become indices or multi-hot vectors, and unknown labels are rejected."""
    assert ClassTargets(["a", "b"], multilabel=False)(["b", "a"]).tolist() == [1, 0]
    assert ClassTargets([0, 1], multilabel=False)(torch.tensor([1, 0])).tolist() == [1, 0]
    assert ClassTargets(["a", "b", "c"], multilabel=True)([["a", "c"], []]).tolist() == [[1, 0, 1], [0, 0, 0]]
    assert ClassTargets([0, 1, 2], multilabel=True)([torch.tensor([0, 2]), np.array([1])]).tolist() == [
        [1, 0, 1],
        [0, 1, 0],
    ]
    with pytest.raises(ValueError):
        ClassTargets(["a", "b"], multilabel=False)(["c"])


def test_datasets_are_picklable_without_the_model(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Datasets can be sent to worker processes, and don't carry the model along."""
    model = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model._initialize_on_labels(["a", "b"])
    dataset = Dataset.from_dict({"text": _TRAIN_TEXTS, "label": ["a", "b"] * 4})
    train, _ = model._create_datasets(dataset["text"], dataset["label"], None, None, 0.25)
    restored = pickle.loads(pickle.dumps(train))
    assert restored.__getitems__([0, 1])[0][0] == train.__getitems__([0, 1])[0][0]
    assert not any(isinstance(value, nn.Module) for value in vars(train).values())
    assert not any(isinstance(value, nn.Module) for value in vars(train.tokenize).values())


@pytest.mark.parametrize("mock_tokenizer", ["wordpiece"], indirect=True)
def test_dataloaders_with_workers_match_main_process(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Across epochs, worker processes load the same batches in the same order as the main process."""
    dataset = Dataset.from_dict(
        {"text": _TRAIN_TEXTS, "other": _TRAIN_TEXTS[::-1], "label": [[float(i)] for i in range(8)]}
    )
    regressor = StaticModelForRegression(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    text_dataset, _ = regressor._create_datasets(dataset["text"], dataset["label"], None, None, 1)
    pairs = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    pair_dataset, _ = pairs._create_pair_datasets(dataset["text"], dataset["other"], None, None, 1)

    for torch_dataset in (text_dataset, pair_dataset):
        epochs = []
        for num_workers in (0, 1):
            torch.manual_seed(0)
            loader = torch_dataset.to_dataloader(shuffle=True, batch_size=2, num_workers=num_workers)
            batches: list[tuple[list, list]] = []
            for _ in range(2):
                batches.extend((x.tolist(), y.tolist()) for x, y in loader)
                torch.rand(1)
            epochs.append(batches)
        assert epochs[0] == epochs[1]


def test_fit_passes_num_workers(
    monkeypatch: pytest.MonkeyPatch, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer
) -> None:
    """num_workers is used for both the training and the validation loader."""
    captured: list[tuple[int, int]] = []

    def fake_run_training_loop(**kwargs: Any) -> dict[str, torch.Tensor]:
        captured.append((kwargs["train_loader"].num_workers, kwargs["val_loader"].num_workers))
        return kwargs["model"].state_dict()

    monkeypatch.setattr("model2vec.train.base.run_training_loop", fake_run_training_loop)
    model = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    model.fit(_TRAIN_TEXTS, _TRAIN_TEXTS, test_size=2, num_workers=3)
    assert captured == [(3, 3)]
