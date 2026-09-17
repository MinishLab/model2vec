import logging
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
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
from model2vec.train.dataset import PairDataset, TextDataset
from model2vec.train.pairs import PairCosineLoss, StaticModelForPairSimilarity
from model2vec.train.regression import StaticModelForRegression
from model2vec.train.similarity import StaticModelForSimilarity
from model2vec.train.trainer import _resolve_max_epochs, resolve_device, run_training_loop
from model2vec.train.utils import get_probable_pad_token_id, logit, seed_everything, train_test_split


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
        vectors=vectors_torched, tokenizer=mock_tokenizer, hidden_dim=256, out_dim=2, n_layers=0, pad_id=0
    )
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]

    head = s.construct_head()
    assert head[0].in_features == mock_vectors.shape[1]


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


def test_textdataset_init() -> None:
    """Test the textdataset init."""
    dataset = TextDataset([[0], [1]], torch.arange(2))
    assert len(dataset) == 2


def test_textdataset_init_incorrect() -> None:
    """Test the textdataset init."""
    with pytest.raises(ValueError):
        TextDataset([[0]], torch.arange(2))


def test_training_batch_padding_is_masked(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Training batches should pad with the model's pad id, so padding stays masked and out of the mean."""
    s = StaticModelForClassification(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer, pad_id=1)
    texts = ["word2", "word2 word3"]

    dataset = s._prepare_dataset(texts, torch.arange(2), max_length=None)
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
    assert s._prepare_dataset(texts, torch.arange(2), max_length=None).tokenized_texts == expected


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
    assert s._prepare_dataset(texts, torch.arange(2), max_length=None).tokenized_texts == expected


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
    dataset = PairDataset([[0], [1]], [[2], [3]])
    assert len(dataset) == 2


def test_pairdataset_init_incorrect() -> None:
    """Test the pair dataset init with mismatched lengths."""
    with pytest.raises(ValueError):
        PairDataset([[0]], [[2], [3]])


def test_pairdataset_collate() -> None:
    """Batches should stack the two padded halves into a single (2, batch, seq_len) tensor."""
    dataset = PairDataset([[1], [1, 2]], [[1, 2, 3], [1]], pad_id=0)
    batch, y = next(iter(dataset.to_dataloader(shuffle=False, batch_size=2)))
    assert batch.shape == (2, 2, 3)
    assert y.shape == (2,)
    assert torch.equal(batch[0], torch.tensor([[1, 0, 0], [1, 2, 0]]))
    assert torch.equal(batch[1], torch.tensor([[1, 2, 3], [1, 0, 0]]))


def test_pairdataset_default_labels_are_positive() -> None:
    """Without explicit labels, every pair defaults to label 1."""
    dataset = PairDataset([[1], [2]], [[3], [4]])
    assert torch.equal(dataset.labels, torch.tensor([1.0, 1.0]))


def test_pairdataset_custom_labels() -> None:
    """Custom labels are stored and returned by the collate function."""
    dataset = PairDataset([[1], [2]], [[3], [4]], labels=[1, 0])
    _, y = next(iter(dataset.to_dataloader(shuffle=False, batch_size=2)))
    assert torch.equal(y, torch.tensor([1.0, 0.0]))


def test_pairdataset_labels_mismatched_length() -> None:
    """Labels must have one entry per pair."""
    with pytest.raises(ValueError):
        PairDataset([[1], [2]], [[3], [4]], labels=[1])


def test_pair_cosine_loss_pushes_towards_label() -> None:
    """Label 1 pairs are pushed towards a cosine similarity of 1, label 0 pairs towards 0."""
    loss_fn = PairCosineLoss()
    out_a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    identical = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    orthogonal = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    assert loss_fn((out_a, identical), torch.tensor([1.0, 1.0])).item() == pytest.approx(0.0, abs=1e-6)
    assert loss_fn((out_a, orthogonal), torch.tensor([1.0, 1.0])).item() == pytest.approx(1.0)
    assert loss_fn((out_a, orthogonal), torch.tensor([0.0, 0.0])).item() == pytest.approx(0.0, abs=1e-6)
    assert loss_fn((out_a, identical), torch.tensor([0.0, 0.0])).item() == pytest.approx(1.0)


def test_pair_similarity_out_dim_defaults_to_embed_dim(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """The output dimension defaults to the input embedding dimension when not specified."""
    s = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    assert s.out_dim == mock_vectors.shape[1]

    s = StaticModelForPairSimilarity(
        vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer, out_dim=7
    )
    assert s.out_dim == 7


def test_pair_similarity_forward(mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity) -> None:
    """The forward pass should return one head output per half of the pair batch."""
    model = mock_trained_pair_similarity_pipeline
    dataset = model._prepare_pair_dataset(["dog cat", "dog"], ["puppy", "kitten cat"], [1, 1], max_length=None)
    batch, _ = next(iter(dataset.to_dataloader(shuffle=False, batch_size=2)))

    with torch.no_grad():
        out_a, out_b = model(batch)
    assert out_a.shape == (2, model.out_dim)
    assert out_b.shape == (2, model.out_dim)


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


def test_pair_similarity_labels_mismatched_length(
    mock_trained_pair_similarity_pipeline: StaticModelForPairSimilarity,
) -> None:
    """Labels must have one entry per training pair, and labels_val one entry per validation pair."""
    with pytest.raises(ValueError):
        mock_trained_pair_similarity_pipeline.fit(["dog", "cat"], ["puppy", "kitten"], labels=[1])
    with pytest.raises(ValueError):
        mock_trained_pair_similarity_pipeline.fit(
            ["dog", "cat"],
            ["puppy", "kitten"],
            text_a_val=["dog"],
            text_b_val=["puppy"],
            labels_val=[1, 0],
        )


def test_pair_similarity_fit_with_explicit_val(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """A model can be fit with explicit validation pairs instead of an automatic split."""
    model = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    text_a = ["word1", "word2", "word3", "word1 word2"]
    text_b = ["word2", "word3", "word1", "word3 word1"]
    labels = [1, 1, 0, 0]
    model.fit(
        text_a,
        text_b,
        labels=labels,
        text_a_val=["word1"],
        text_b_val=["word2"],
        labels_val=[1],
        early_stopping_patience=1,
        max_epochs=1,
    )


def test_pair_similarity_fit_with_labels(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """A model can be fit with a mix of positive and negative pair labels."""
    model = StaticModelForPairSimilarity(vectors=torch.from_numpy(mock_vectors).float(), tokenizer=mock_tokenizer)
    text_a = ["word1", "word2", "word3", "word1 word2"]
    text_b = ["word2", "word3", "word1", "word3 word1"]
    labels = [1, 1, 0, 0]
    model.fit(text_a, text_b, labels=labels, early_stopping_patience=1, max_epochs=1)


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


def test_train_test_split() -> None:
    """Test the train test split function."""
    a, b, c, d = train_test_split(["0", "1", "2", "3"], ["1", "1", "0", "0"], 0.5)
    assert len(a) == 2
    assert len(b) == 2
    assert len(c) == len(a)
    assert len(d) == len(b)


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


def test_determine_class_weight(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test what the class weights are."""
    w_dict = dict(zip(mock_trained_pipeline.classes, [0.5, 3]))
    c1, c2 = mock_trained_pipeline.classes_
    y: list[str] | list[list[str]]
    if mock_trained_pipeline.multilabel:
        y = [*[[c1]] * 100, *[[c2]] * 50]
    else:
        y = [*[c1] * 100, *[c2] * 50]
    w = mock_trained_pipeline._determine_class_weight(w_dict, y)
    assert isinstance(w, torch.Tensor)
    assert w.tolist() == [0.5, 3]

    w = mock_trained_pipeline._determine_class_weight(w_dict, y)
    assert isinstance(w, torch.Tensor)
    assert w.tolist() == [0.5, 3]

    w = mock_trained_pipeline._determine_class_weight("balanced", y)
    assert isinstance(w, torch.Tensor)
    assert w.tolist() == [0.75, 1.5]


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
        compute_metrics=lambda head_out, y, loss: {"val_loss": 1.0},
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
