from dataclasses import dataclass


@dataclass
class Token:
    """A class to represent a token."""

    # The surface form: used for featurizing
    form: str
    # The normalized form: preprocessed by the new tokenizer
    normalized_form: str
    # Whether the word is a continuing subword.
    is_subword: bool
    # Whether the token is internal to the model.
    is_internal: bool
    # Whether the token is a multiword token
    is_multiword: bool = False
