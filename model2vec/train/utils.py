import logging

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

_KNOWN_PAD_TOKENS = ("[PAD]", "<pad>")


def get_probable_pad_token_id(tokenizer: Tokenizer) -> int:
    """Get a probable pad token by using the padding module and falling back to guessing."""
    if tokenizer.padding is not None:
        return tokenizer.padding["pad_id"]
    vocab = tokenizer.get_vocab()
    for token in _KNOWN_PAD_TOKENS:
        token_id = vocab.get(token)
        if token_id is not None:
            return token_id

    logger.warning("No known pad token found, using 0 as default")
    return 0
