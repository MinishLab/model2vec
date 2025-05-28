from __future__ import annotations

from typing import Any

import numpy as np


def process_tokenizer(
    tokenizer_json: dict[str, Any], pre_tokenized_tokens: list[str], unk_token: str | None
) -> dict[str, Any]:
    """Process the WordPiece tokenizer JSON."""
    if tokenizer_json["model"]["type"] == "Unigram":
        return _process_unigram(tokenizer_json, pre_tokenized_tokens, unk_token)
    tokenizer_json["model"]["type"] = "Unigram"
    tokenizer_json["model"]["unk_id"] = pre_tokenized_tokens.index(unk_token) if unk_token else None

    token_weights = np.asarray([_calculate_token_weight_for_unigram(token) for token in pre_tokenized_tokens])
    proba = (token_weights / np.sum(token_weights)).tolist()
    tokenizer_json["model"]["vocab"] = [(token, np.log(p)) for token, p in zip(pre_tokenized_tokens, proba)]

    return tokenizer_json


def _process_unigram(
    tokenizer_json: dict[str, Any], pre_tokenized_tokens: list[str], unk_token: str | None
) -> dict[str, Any]:
    """Process the Unigram tokenizer JSON."""
    current_probas = dict(tokenizer_json["model"]["vocab"])
    avg_proba = sum(current_probas.values()) / len(current_probas)
    new_probas = [[word, current_probas.get(word, avg_proba)] for word in pre_tokenized_tokens]
    tokenizer_json["model"]["vocab"] = new_probas

    tokens, _ = zip(*tokenizer_json["model"]["vocab"])
    if unk_token is not None:
        tokenizer_json["model"]["unk_id"] = list(tokens).index(unk_token)

    return tokenizer_json


def _calculate_token_weight_for_unigram(token: str) -> float:
    """Calculate the token weight for Unigram."""
    # Always prefer longer tokens.
    return len(token) + token.count("▁") + token.count("Ġ")
