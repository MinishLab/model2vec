import logging
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from skeletoken import TokenizerModel
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from model2vec.model import StaticModel
from model2vec.train import StaticModelForClassification
from model2vec.train.base import BaseFinetuneable
from model2vec.train.dataset import TextDataset
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
    assert torch.all(w == logit(torch.ones(len(mock_vectors))))


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


def test_textdataset_init() -> None:
    """Test the textdataset init."""
    dataset = TextDataset([[0], [1]], torch.arange(2))
    assert len(dataset) == 2


def test_textdataset_init_incorrect() -> None:
    """Test the textdataset init."""
    with pytest.raises(ValueError):
        TextDataset([[0]], torch.arange(2))


def test_predict(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the predict function."""
    result = mock_trained_pipeline.predict(["dog cat", "dog"]).tolist()
    if mock_trained_pipeline.multilabel:
        if type(mock_trained_pipeline.classes_[0]) == str:
            assert result == [["a", "b"], ["a", "b"]]
        else:
            assert result == [[0, 1], [0, 1]]
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
