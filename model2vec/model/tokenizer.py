from __future__ import annotations

from typing import Iterator, Protocol, TypeVar

from ahocorasick import Automaton
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


class AutomatonProtocol(Protocol):
    def __init__(self, *args, **kwargs) -> None: ...

    def add_word(self, word: str, value: str) -> None: ...

    def make_automaton(self) -> None: ...

    def iter_long(self, text: str) -> Iterator[tuple[int, str]]: ...


class Model2VecTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    @classmethod
    def from_vocab(
        cls: type[TokenizerType], vocabulary: dict[str, int], unk_token: str, pad_token: str
    ) -> TokenizerType:
        tokenizer = _create_hf_tokenizer_from_vocab(vocabulary, unk_token, pad_token)

        return cls(tokenizer)


TokenizerType = TypeVar("TokenizerType", bound=Model2VecTokenizer)


class MultiwordModel2VecTokenizer(Model2VecTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, automaton: AutomatonProtocol, delimiter: str) -> None:
        super().__init__(tokenizer)
        self.automaton = automaton
        self.delimiter = delimiter
        self.pre_tokenizer: Whitespace = self.tokenizer.backend_tokenizer.pre_tokenizer
        self.normalizer: Sequence = self.tokenizer.backend_tokenizer.normalizer

    def _pseudo_tokenize(self, text: str) -> str:
        normalized = self.normalizer.normalize_str(text)
        tokens, _ = zip(*self.pre_tokenizer.pre_tokenize_str(normalized))

        return f"{self.delimiter}{self.delimiter.join(tokens)}{self.delimiter}"

    def tokenize(self, text: str) -> list[str]:
        delimited = self._pseudo_tokenize(text)

        tokens = []
        for _, token in self.automaton.iter_long(delimited):
            tokens.append(token)
        return tokens

    @classmethod
    def from_vocab(
        cls: type[MultiwordModel2VecTokenizer],
        vocabulary: dict[str, int],
        unk_token: str,
        pad_token: str,
        delimiter: str = "   ",
    ) -> MultiwordModel2VecTokenizer:
        tokenizer = _create_hf_tokenizer_from_vocab(vocabulary, unk_token, pad_token)
        automaton: AutomatonProtocol = Automaton()
        for token in tokenizer.vocab:
            token = token.replace(" ", delimiter)
            automaton.add_word(f" {token} ", token)

        automaton.make_automaton()

        return cls(tokenizer, automaton, delimiter)


def create_model2vec_tokenizer(vocabulary: dict[str, int], unk_token: str, pad_token: str) -> Model2VecTokenizer:
    if any(" " in token for token in vocabulary):
        return MultiwordModel2VecTokenizer.from_vocab(vocabulary, unk_token, pad_token)
    return Model2VecTokenizer.from_vocab(vocabulary, unk_token, pad_token)


def _create_hf_tokenizer_from_vocab(
    vocabulary: dict[str, int], unk_token: str, pad_token: str
) -> PreTrainedTokenizerFast:
    """
    Creates a _word_ level tokenizer from a pre-specified vocabulary and an unk token.

    This tokenizer automatically normalizes, and splits using the Whitespace splitter.

    :param vocabulary: The vocabulary mapping from strings to integers.
    :param unk_token: The string representing the unk token.
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
