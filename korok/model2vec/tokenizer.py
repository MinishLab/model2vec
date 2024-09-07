from __future__ import annotations

from typing import TypeVar

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


class Model2VecTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        """Initialize the Model2VecTokenizer."""
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[str]:
        """Tokenize the text."""
        return self.tokenizer.tokenize(text)

    @classmethod
    def from_vocab(
        cls: type[TokenizerType], vocabulary: dict[str, int], unk_token: str, pad_token: str
    ) -> TokenizerType:
        """
        Create a Model2VecTokenizer from a vocabulary.

        :param vocabulary: The vocabulary mapping from strings to integers.
        :param unk_token: The string representing the unk token.
        :param pad_token: The string representing the pad token.
        :return: A Model2VecTokenizer.
        """
        tokenizer = _create_hf_tokenizer_from_vocab(vocabulary, unk_token, pad_token)

        return cls(tokenizer)


TokenizerType = TypeVar("TokenizerType", bound=Model2VecTokenizer)


def create_model2vec_tokenizer_from_vocab(
    vocabulary: dict[str, int], unk_token: str, pad_token: str
) -> Model2VecTokenizer:
    """
    Create a Model2VecTokenizer from a vocabulary.

    :param vocabulary: The vocabulary mapping from strings to integers.
    :param unk_token: The string representing the unk token.
    :param pad_token: The string representing the pad token.
    :return: A Model2VecTokenizer.
    """
    return Model2VecTokenizer.from_vocab(vocabulary, unk_token, pad_token)


def _create_hf_tokenizer_from_vocab(
    vocabulary: dict[str, int], unk_token: str, pad_token: str
) -> PreTrainedTokenizerFast:
    """
    Creates a _word_ level tokenizer from a pre-specified vocabulary and an unk token.

    This tokenizer automatically normalizes, and splits using the Whitespace splitter.

    :param vocabulary: The vocabulary mapping from strings to integers.
    :param unk_token: The string representing the unk token.
    :param pad_token: The string representing the pad token.
    :raises: ValueError if the unk token is not in the vocabulary.
    :return: A word-level tokenizer.
    """
    if unk_token not in vocabulary:
        raise ValueError(f"unk token: {unk_token} was not in your vocabulary.")

    model = WordLevel(vocab=vocabulary, unk_token=unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()

    # NOTE: unk token can be cased, doesn't matter.
    is_cased = any(token != token.lower() for token in vocabulary if token not in {unk_token, pad_token})

    normalization_steps = [NFKC()]
    if not is_cased:
        normalization_steps.append(Lowercase())
    tokenizer.normalizer = Sequence(normalization_steps)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({"unk_token": unk_token, "pad_token": pad_token})

    return tokenizer
