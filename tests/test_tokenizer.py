import json

import pytest
from transformers import PreTrainedTokenizerFast

from model2vec.tokenizer.model import _calculate_token_weight_for_unigram, _process_unigram, process_tokenizer
from model2vec.tokenizer.normalizer import replace_normalizer
from model2vec.tokenizer.pretokenizer import _FORBIDDEN_PRETOKENIZERS, _fix_single_pretokenizer, replace_pretokenizer
from model2vec.tokenizer.tokenizer import _rename_added_token, create_tokenizer


def test_fix_single_pretokenizer() -> None:
    """Test the _fix_single_pretokenizer function."""
    result = _fix_single_pretokenizer({"type": "ByteLevel", "add_prefix_space": False, "use_regex": True})
    assert result == {"type": "ByteLevel", "add_prefix_space": True, "use_regex": False}

    for tokenizer_type in _FORBIDDEN_PRETOKENIZERS:
        result = _fix_single_pretokenizer({"type": tokenizer_type})
        assert result is None

    result = _fix_single_pretokenizer(
        {"type": "Metaspace", "split": True, "prepend_scheme": "never", "replacement": "▁"}
    )
    assert result == {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": False}


def test_replace_pretokenizer(mock_berttokenizer: PreTrainedTokenizerFast) -> None:
    """Test the replace_pretokenizer function."""
    tokenizer = replace_pretokenizer(mock_berttokenizer.backend_tokenizer)
    assert tokenizer.pre_tokenizer is not None
    assert tokenizer.pre_tokenizer.__class__.__name__ == "Metaspace"
    assert tokenizer.pre_tokenizer.replacement == "▁"
    assert tokenizer.pre_tokenizer.prepend_scheme == "always"
    assert not tokenizer.pre_tokenizer.split

    tokenizer.pre_tokenizer = None  # type: ignore
    tokenizer = replace_pretokenizer(tokenizer)
    assert tokenizer.pre_tokenizer is not None
    assert tokenizer.pre_tokenizer.__class__.__name__ == "Metaspace"
    assert tokenizer.pre_tokenizer.replacement == "▁"
    assert tokenizer.pre_tokenizer.prepend_scheme == "always"
    assert tokenizer.pre_tokenizer.split is False


def test_replace_normalizer(mock_berttokenizer: PreTrainedTokenizerFast) -> None:
    """Test the replace_normalizer function."""
    tokenizer = replace_normalizer(mock_berttokenizer.backend_tokenizer)
    assert tokenizer.normalizer is not None
    assert tokenizer.normalizer.__class__.__name__ == "Sequence"

    assert tokenizer.normalizer.normalize_str("Hello, World!") == "hello , world !"

    tokenizer.normalizer = None  # type: ignore
    tokenizer = replace_normalizer(tokenizer)
    assert tokenizer.normalizer.normalize_str("Hello, World!") == "Hello , World !"


@pytest.mark.parametrize(
    "word,weight",
    [
        ("dog", 3),
        ("cat", 3),
        ("▁longer▁word", 14),
        ("▁word", 6),
        ("▁", 2),  # Single underscore
        ("", 0),  # Empty string
        ("▁a" * 100, 300),  # Long word with underscores
    ],
)
def test_calculate_token_weight_for_unigram(word: str, weight: int) -> None:
    """Test the _calculate_token_weight_for_unigram function."""
    assert _calculate_token_weight_for_unigram(word) == weight


def test_process_tokenizer(mock_berttokenizer: PreTrainedTokenizerFast) -> None:
    """Test the process_tokenizer function."""
    vocab = ["dog", "cat", "longer_word", "word", "a" * 100, "[UNK]"]
    tokenizer_json = json.loads(mock_berttokenizer.backend_tokenizer.to_str())
    tokenizer_json = process_tokenizer(tokenizer_json=tokenizer_json, pre_tokenized_tokens=vocab, unk_token="[UNK]")

    assert tokenizer_json["model"]["type"] == "Unigram"
    assert tokenizer_json["model"]["unk_id"] == 5  # Index of "[UNK]"
    assert len(tokenizer_json["model"]["vocab"]) == 6
    assert all(isinstance(token, tuple) and len(token) == 2 for token in tokenizer_json["model"]["vocab"])
    for (x, _), y in zip(tokenizer_json["model"]["vocab"], vocab):
        assert x == y, f"Expected {y}, but got {x}"


def test_process_unigram() -> None:
    """Test the _process_unigram function."""
    vocab = ["dog", "cat", "longer_word", "word", "a" * 100, "[UNK]"]
    orig_vocab = [("dog", 0), ("cat", 0)]
    model = {"model": {"type": "Unigram", "vocab": orig_vocab}}
    processed_model = _process_unigram(model, vocab, "[UNK]")
    assert processed_model["model"]["type"] == "Unigram"
    assert processed_model["model"]["unk_id"] == 5  # Index of "[UNK]"
    assert len(processed_model["model"]["vocab"]) == 6
    assert all(isinstance(token, list) and len(token) == 2 for token in processed_model["model"]["vocab"])

    for (x, score), y in zip(processed_model["model"]["vocab"], vocab):
        assert x == y, f"Expected {y}, but got {x}"
        if x in orig_vocab:
            assert score == 0

    assert process_tokenizer(model, vocab, "[UNK]") == processed_model


def test_rename_added_token() -> None:
    """Test the _rename_added_token function."""
    # Invalid input
    result = _rename_added_token(None, "a", [{"content": "a", "id": 0}], ["a"])
    assert result == [{"content": "a", "id": 0}]

    # Rename 'a' to 'c'
    result = _rename_added_token("a", "c", [{"content": "a"}], ["a"])
    assert result == [{"content": "c", "id": 0}]


def test_create_tokenizer(mock_berttokenizer: PreTrainedTokenizerFast) -> None:
    """Test the create_tokenizer function."""
    tokenizer = create_tokenizer(tokenizer=mock_berttokenizer, vocabulary=["dog", "catssssss"], token_remove_regex=None)
    assert tokenizer.backend_tokenizer.get_vocab_size() == 29525
    assert tokenizer.encode("catssssss") == [29524]
