import numpy as np
import pytest
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as SentenceTransformerModels
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel


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
def mock_transformer() -> BertModel:
    """Create a mock transformer."""
    # Define the configuration
    config = BertConfig(
        hidden_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=2,
    )

    # Initialize a BertModel with random weights
    random_bert = BertModel(config)

    # Create a custom transformer class to bypass model loading behavior
    class MockTransformer(SentenceTransformerModels.Transformer):
        def __init__(self, model: BertModel) -> None:
            # Skip loading from a name or path, use the pre-loaded model
            self.auto_model = model

    # Create a transformer model and a pooling model
    word_embedding_model = MockTransformer(random_bert)
    pooling_model = SentenceTransformerModels.Pooling(word_embedding_model.get_word_embedding_dimension())

    # Initialize the SentenceTransformer
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


@pytest.fixture
def mock_vectors() -> np.ndarray:
    """Create mock vectors."""
    return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]])


@pytest.fixture
def mock_config() -> dict[str, str]:
    """Create a mock config."""
    return {"some_config": "value"}
