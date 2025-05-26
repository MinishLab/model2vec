from dataclasses import dataclass


@dataclass
class Token:
    """A class to represent a token."""

    form: str
    # The normalized and pretokenized form of the token
    normalized_form: str
    # Whether the word is a continuing subword.
    is_subword: bool
    # Whether the token is internal to the model.
    is_internal: bool
