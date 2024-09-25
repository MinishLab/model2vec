import numpy as np
import pytest
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as SentenceTransformerModels
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast


@pytest.fixture
def mock_tokenizer() -> PreTrainedTokenizerFast:
    """Create a mock tokenizer."""
    # Define the vocabulary and special tokens
    vocab = ["word1", "word2", "word3", "[UNK]", "[PAD]"]
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    unk_token = "[UNK]"

    # Create a WordLevel model with the vocab and set the unk_token
    tokenizer_model = models.WordLevel(vocab=vocab_dict, unk_token=unk_token)

    # Initialize the tokenizer with the WordLevel model
    tokenizer = Tokenizer(tokenizer_model)

    # Add basic components to handle text preprocessing
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Add a simple decoder
    tokenizer.decoder = decoders.WordPiece()
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
