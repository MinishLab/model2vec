from __future__ import annotations

import json
from typing import Any

from tokenizers import Tokenizer

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


def replace_pretokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """Fixes a single pretokenizer to allow multiword units."""
    tokenizer_json = json.loads(tokenizer.to_str())
    pre_tokenizer_json = tokenizer_json.get("pre_tokenizer", None)

    if pre_tokenizer_json is None:
        pre_tokenizer_json = _BASIC_METASPACE

    elif pre_tokenizer_json["type"] == "Sequence":
        new_pretokenizers = []
        for single_pretokenizer in pre_tokenizer_json["pretokenizers"]:
            new_pretokenizer = _fix_single_pretokenizer(single_pretokenizer)
            if new_pretokenizer is not None:
                new_pretokenizers.append(new_pretokenizer)

        if new_pretokenizers:
            pre_tokenizer_json["pretokenizers"] = new_pretokenizers
        else:
            pre_tokenizer_json = _BASIC_METASPACE

    pre_tokenizer_json = _fix_single_pretokenizer(pre_tokenizer_json) or _BASIC_METASPACE
    tokenizer_json["pre_tokenizer"] = pre_tokenizer_json

    return tokenizer.from_str(json.dumps(tokenizer_json))
