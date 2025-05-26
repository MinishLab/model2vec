from string import punctuation

from tokenizers import Regex, Tokenizer
from tokenizers.normalizers import Normalizer, Replace, Sequence, Strip


def replace_normalizer(
    tokenizer: Tokenizer,
) -> Tokenizer:
    """
    Replace the normalizer for the tokenizer.

    The new normalizer will replace punctuation with a space before and after the punctuation.
    It will also replace multiple spaces with a single space and strip the right side of the string.
    If the tokenizer already has a normalizer, it will be added to the new normalizer.
    If the tokenizer does not have a normalizer, a new normalizer will be created.

    :param tokenizer: The tokenizer to change.
    :return: The tokenizer with a replaced normalizer.
    """
    normalizer = tokenizer.normalizer
    new_normalizers = []
    for char in punctuation:
        new_normalizers.append(Replace(char, f" {char} "))

    new_normalizers.append(Replace(Regex(r"\s+"), " "))
    new_normalizers.append(Strip(right=True))
    if normalizer is None:
        normalizer = Sequence(new_normalizers)  # type: ignore
    else:
        normalizer = Sequence([normalizer] + new_normalizers)  # type: ignore
    tokenizer.normalizer = normalizer  # type: ignore

    return tokenizer
