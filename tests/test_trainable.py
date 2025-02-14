from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from model2vec.model import StaticModel
from model2vec.train import StaticModelForClassification
from model2vec.train.base import FinetunableStaticModel, TextDataset


@pytest.mark.parametrize("n_layers", [0, 1, 2, 3])
def test_init_predict(n_layers: int, mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test successful initialization of StaticModelForClassification."""
    vectors_torched = torch.from_numpy(mock_vectors)
    s = StaticModelForClassification(vectors=vectors_torched, tokenizer=mock_tokenizer, n_layers=n_layers)
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]
    assert s.classes == s.classes_
    assert s.classes == ["0", "1"]

    head = s.construct_head()
    assert head[0].in_features == mock_vectors.shape[1]
    head = s.construct_head()
    assert head[0].in_features == mock_vectors.shape[1]
    assert head[-1].out_features == 2


def test_init_base_class(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test successful initialization of the base class."""
    vectors_torched = torch.from_numpy(mock_vectors)
    s = FinetunableStaticModel(vectors=vectors_torched, tokenizer=mock_tokenizer)
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]

    head = s.construct_head()
    assert head[0].in_features == mock_vectors.shape[1]


def test_init_base_from_model(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test initializion from a static model."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    s = FinetunableStaticModel.from_static_model(model)
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]

    with TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        s = FinetunableStaticModel.from_pretrained(model_name=temp_dir)
        assert s.vectors.shape == mock_vectors.shape
        assert s.w.shape[0] == mock_vectors.shape[0]


def test_init_classifier_from_model(mock_vectors: np.ndarray, mock_tokenizer: Tokenizer) -> None:
    """Test initializion from a static model."""
    model = StaticModel(vectors=mock_vectors, tokenizer=mock_tokenizer)
    s = StaticModelForClassification.from_static_model(model)
    assert s.vectors.shape == mock_vectors.shape
    assert s.w.shape[0] == mock_vectors.shape[0]

    with TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        s = StaticModelForClassification.from_pretrained(model_name=temp_dir)
        assert s.vectors.shape == mock_vectors.shape
        assert s.w.shape[0] == mock_vectors.shape[0]


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
        result_1 = mock_trained_pipeline._encode(torch.tensor([[0, 1], [1, 0]]).long()).numpy()
    result_2 = staticmodel.embedding[[[0, 1], [1, 0]]].mean(0)
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
        assert result == [["a", "b"], ["a", "b"]]
    else:
        assert result == ["b", "b"]


def test_predict_proba(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Test the predict function."""
    result = mock_trained_pipeline.predict_proba(["dog cat", "dog"])
    assert result.shape == (2, 2)


def test_convert_to_pipeline(mock_trained_pipeline: StaticModelForClassification) -> None:
    """Convert a model to a pipeline."""
    mock_trained_pipeline.eval()
    pipeline = mock_trained_pipeline.to_pipeline()
    encoded_pipeline = pipeline.model.encode(["dog cat", "dog"])
    encoded_model = mock_trained_pipeline(mock_trained_pipeline.tokenize(["dog cat", "dog"]))[1].detach().numpy()
    assert np.allclose(encoded_pipeline, encoded_model)
    a = pipeline.predict(["dog cat", "dog"]).tolist()
    b = mock_trained_pipeline.predict(["dog cat", "dog"]).tolist()
    assert a == b
    p1 = pipeline.predict_proba(["dog cat", "dog"])
    p2 = mock_trained_pipeline.predict_proba(["dog cat", "dog"])
    assert np.allclose(p1, p2)


def test_train_test_split() -> None:
    """Test the train test split function."""
    a, b, c, d = StaticModelForClassification._train_test_split(["0", "1", "2", "3"], ["1", "1", "0", "0"], 0.5)
    assert len(a) == 2
    assert len(b) == 2
    assert len(c) == len(a)
    assert len(d) == len(b)
