from string import punctuation

from tokenizers import Regex
from tokenizers.normalizers import Normalizer, Replace, Sequence, Strip


def prepare_normalizer(
    normalizer: Normalizer,
) -> Normalizer:
    """
    Prepare the normalizer for the tokenizer.

    This function sets the normalizer for the tokenizer based on the provided normalizer type.
    If no normalizer is provided, it uses the default one.

    :param normalizer: The tokenizer to prepare.
    :return: The prepared tokenizer.
    """
    new_normalizers = []
    for char in punctuation:
        new_normalizers.append(Replace(char, f" {char} "))

    new_normalizers.append(Replace(Regex(r"\s+"), " "))
    new_normalizers.append(Strip(right=True))
    if normalizer is None:
        return Sequence(new_normalizers)

    return Sequence([normalizer] + new_normalizers)  # type: ignore
