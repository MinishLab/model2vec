from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from model2vec.inference import StaticModelPipeline
from model2vec.train import StaticModelForClassification

_TOKENIZER_TYPES = ["wordpiece", "bpe", "unigram"]


@pytest.fixture(scope="session", params=_TOKENIZER_TYPES, ids=_TOKENIZER_TYPES)
def mock_tokenizer(request: pytest.FixtureRequest) -> Tokenizer:
    """Create a mock tokenizer."""
    vocab = ["[PAD]", "word1", "word2", "word3", "[UNK]"]
    unk_token = "[UNK]"

    tokenizer_type = request.param

    if tokenizer_type == "wordpiece":
        model = WordPiece(
            vocab={token: idx for idx, token in enumerate(vocab)}, unk_token=unk_token, max_input_chars_per_word=100
        )
    elif tokenizer_type == "bpe":
        model = BPE(
            vocab={token: idx for idx, token in enumerate(vocab)},
            merges=[],
            unk_token=unk_token,
            fuse_unk=True,
            ignore_merges=True,
        )
    elif tokenizer_type == "unigram":
        model = Unigram(vocab=[(token, 0.0) for token in vocab], unk_id=0, byte_fallback=False)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore  # Tokenizer issue

    return tokenizer


@pytest.fixture(scope="function")
def mock_berttokenizer() -> PreTrainedTokenizerFast:
    """Load the real BertTokenizerFast from the provided tokenizer.json file."""
    return cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained("tests/data/test_tokenizer"))


@pytest.fixture
def mock_transformer() -> AutoModel:
    """Create a mock transformer model."""

    class MockPreTrainedModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.name_or_path = "mock-model"

        def to(self, device: str) -> MockPreTrainedModel:
            self.device = device
            return self

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            # Simulate a last_hidden_state output for a transformer model
            batch_size, seq_length = kwargs["input_ids"].shape
            # Return a tensor of shape (batch_size, seq_length, 768)
            return type(
                "BaseModelOutputWithPoolingAndCrossAttentions",
                (object,),
                {
                    "last_hidden_state": torch.rand(batch_size, seq_length, 768)  # Simulate 768 hidden units
                },
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            # Simply call the forward method to simulate the same behavior as transformers models
            return self.forward(*args, **kwargs)

    return MockPreTrainedModel()


@pytest.fixture(scope="session")
def mock_vectors() -> np.ndarray:
    """Create mock vectors."""
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def mock_config() -> dict[str, str]:
    """Create a mock config."""
    return {"some_config": "value"}


@pytest.fixture(scope="session")
def mock_inference_pipeline(mock_trained_pipeline: StaticModelForClassification) -> StaticModelPipeline:
    """Mock pipeline."""
    return mock_trained_pipeline.to_pipeline()


@pytest.fixture(
    params=[
        (False, "single_label", "str"),
        (False, "single_label", "int"),
        (True, "multilabel", "str"),
        (True, "multilabel", "int"),
    ],
    ids=lambda param: f"{param[1]}_{param[2]}",
    scope="session",
)
def mock_trained_pipeline(request: pytest.FixtureRequest) -> StaticModelForClassification:
    """Mock StaticModelForClassification with different label formats."""
    tokenizer = AutoTokenizer.from_pretrained("tests/data/test_tokenizer").backend_tokenizer
    torch.random.manual_seed(42)
    vectors_torched = torch.randn(len(tokenizer.get_vocab()), 12)
    model = StaticModelForClassification(vectors=vectors_torched, tokenizer=tokenizer, hidden_dim=12).to("cpu")

    X = ["dog", "cat"]
    is_multilabel, label_type = request.param[0], request.param[2]

    if label_type == "str":
        y = [["a", "b"], ["a"]] if is_multilabel else ["a", "b"]  # type: ignore
    else:
        y = [[0, 1], [0]] if is_multilabel else [0, 1]  # type: ignore

    model.fit(X, y)

    return model
