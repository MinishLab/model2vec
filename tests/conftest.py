import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


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
def mock_vectors() -> np.ndarray:
    """Create mock vectors."""
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def mock_config() -> dict[str, str]:
    """Create a mock config."""
    return {"some_config": "value"}
