from typing import Any

_FORBIDDEN_PRETOKENIZERS = (
    "WhiteSpace",
    "WhitespaceSplit",
    "BertPreTokenizer",
    "CharDelimiterSplit",
    "Punctuation",
    "Split",
    "UnicodeScripts",
)
_BASIC_METASPACE = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": False}


def _fix_single_pretokenizer(pre_tokenizer: dict[str, Any]) -> dict[str, Any] | None:
    """Fixes a single pretokenizer to allow multiword units."""
    if pre_tokenizer["type"] in _FORBIDDEN_PRETOKENIZERS:
        return None
    if pre_tokenizer["type"] == "ByteLevel":
        pre_tokenizer["add_prefix_space"] = True
        pre_tokenizer["use_regex"] = False
    if pre_tokenizer["type"] == "Metaspace":
        pre_tokenizer["split"] = False
        pre_tokenizer["prepend_scheme"] = "always"

    return pre_tokenizer


def fix_pretokenizer(pretokenizer: dict[str, Any] | None) -> dict[str, Any]:
    """Fixes a single pretokenizer to allow multiword units."""
    if pretokenizer is None:
        return _BASIC_METASPACE

    if pretokenizer["type"] == "Sequence":
        new_pretokenizers = []
        for single_pretokenizer in pretokenizer["pretokenizers"]:
            new_pretokenizer = _fix_single_pretokenizer(single_pretokenizer)
            if new_pretokenizer is not None:
                new_pretokenizers.append(new_pretokenizer)
        pretokenizer["pretokenizers"] = new_pretokenizers

        if not pretokenizer:
            return _BASIC_METASPACE

        return pretokenizer

    single_pretokenizer = _fix_single_pretokenizer(pretokenizer)
    if single_pretokenizer is None:
        return _BASIC_METASPACE

    return single_pretokenizer
