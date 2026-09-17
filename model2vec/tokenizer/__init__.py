from model2vec.utils import get_package_extras, importable

importable("transformers", "tokenizer")
_REQUIRED_EXTRA = "tokenizer"

for extra_dependency in get_package_extras("model2vec", _REQUIRED_EXTRA):
    importable(extra_dependency, _REQUIRED_EXTRA)

from model2vec.tokenizer.tokenizer import (  # noqa: E402
    add_vocabulary_to_model,
    clean_and_create_vocabulary,
    prune_vocabulary,
    turn_tokens_into_ids,
)

__all__ = ["add_vocabulary_to_model", "clean_and_create_vocabulary", "prune_vocabulary", "turn_tokens_into_ids"]
