import numpy as np
import pytest
from transformers import PreTrainedTokenizerFast

from model2vec.distill.tokenizer import create_tokenizer_from_vocab


@pytest.fixture
def mock_tokenizer() -> PreTrainedTokenizerFast:
    """Create a mock tokenizer."""
    vocab = ["word1", "word2", "word3", "[UNK]", "[PAD]"]
    unk_token = "[UNK]"
    pad_token = "[PAD]"

    tokenizer = create_tokenizer_from_vocab(vocab, unk_token, pad_token)

    return tokenizer


@pytest.fixture
def mock_vectors() -> np.ndarray:
    """Create mock vectors."""
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def mock_config() -> dict[str, str]:
    """Create a mock config."""
    return {"some_config": "value"}
