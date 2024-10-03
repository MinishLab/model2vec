from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, AutoTokenizer


@pytest.fixture
def mock_tokenizer() -> Tokenizer:
    """Create a mock tokenizer."""
    vocab = ["word1", "word2", "word3", "[UNK]", "[PAD]"]
    unk_token = "[UNK]"

    model = WordLevel(vocab={word: idx for idx, word in enumerate(vocab)}, unk_token=unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()

    return tokenizer


@pytest.fixture
def mock_berttokenizer() -> AutoTokenizer:
    """Load the real BertTokenizerFast from the provided tokenizer.json file."""
    return AutoTokenizer.from_pretrained("tests/data/test_tokenizer")


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

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
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

    return MockPreTrainedModel()


@pytest.fixture
def mock_vectors() -> np.ndarray:
    """Create mock vectors."""
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def mock_config() -> dict[str, str]:
    """Create a mock config."""
    return {"some_config": "value"}
