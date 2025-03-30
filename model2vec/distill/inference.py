# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import Protocol, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from model2vec.distill.tokenizer import get_unk_token
from model2vec.distill.utils import filter_vocabulary_by_regex

logger = logging.getLogger(__name__)


PathLike = Union[Path, str]

_DEFAULT_BATCH_SIZE = 1024


class ModulewithWeights(Protocol):
    weight: torch.nn.Parameter


def create_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    use_subword: bool,
    tokens: list[str],
    device: str,
    token_remove_regex: re.Pattern | None,
) -> tuple[list[str], np.ndarray]:
    """
    Create output embeddings for a bunch of tokens using a pretrained model.

    It does a forward pass for all tokens passed in `tokens`.

    :param model: The model to use.
        This should be a transformers model.
    :param tokenizer: The tokenizer to use.
    :param use_subword: Whether to include subword tokens in the output.
    :param tokens: The tokens to use.
    :param device: The torch device to use.
    :param token_remove_regex: A regex pattern to remove tokens from the vocabulary.
    :return: The tokens and output embeddings.
    """
    model = model.to(device)

    out_weights: np.ndarray
    intermediate_weights: list[np.ndarray] = []

    pad_token = tokenizer.pad_token_type_id

    out_tokens = []
    tokenized: list[torch.Tensor] = []

    if use_subword:
        if token_remove_regex is not None:
            # Sort the vocabulary by id, important for zipf.
            sorted_vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
            id_list = filter_vocabulary_by_regex(token_remove_regex, sorted_vocab)
            ids = torch.Tensor(id_list).long()
        else:
            # If the token remove regex is None, just use all tokens.
            ids = torch.arange(len(tokenizer.get_vocab()))

    elif unk_token := get_unk_token(tokenizer.backend_tokenizer):
        # Include unk token. This is necessary for some models.
        ids = torch.Tensor(tokenizer.convert_tokens_to_ids([unk_token])).long()
    else:
        ids = None

    if ids is not None:
        dummy_encoding = tokenizer.encode("A")
        bos_token_id, eos_token_id = dummy_encoding[0], dummy_encoding[-1]

        bos = torch.full([len(ids)], fill_value=bos_token_id)
        eos = torch.full([len(ids)], fill_value=eos_token_id)

        tokenized.extend(torch.stack([bos, ids, eos], dim=1))
        out_tokens.extend(tokenizer.convert_ids_to_tokens(ids))

    tokenized.extend([tokenizer.encode_plus(token, return_tensors="pt")["input_ids"][0] for token in tokens])

    for batch_idx in tqdm(range(0, len(tokenized), _DEFAULT_BATCH_SIZE)):
        batch = tokenized[batch_idx : batch_idx + _DEFAULT_BATCH_SIZE]

        encoded = {}
        encoded["input_ids"] = pad_sequence(batch, batch_first=True, padding_value=pad_token)
        encoded["attention_mask"] = encoded["input_ids"] != pad_token

        # Add token_type_ids only if the model supports it
        if "token_type_ids" in inspect.getfullargspec(model.forward).args:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

        out = _encode_mean_using_model(model, encoded)
        intermediate_weights.append(out.numpy())

    out_tokens.extend(tokens)
    out_weights = np.concatenate(intermediate_weights)

    return out_tokens, out_weights


@torch.no_grad()
def _encode_mean_using_model(model: PreTrainedModel, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Encode a batch of tokens using a model.

    Note that if a token in the input batch does not have any embeddings, it will be output as a vector of zeros.
    So detection of these is necessary.

    :param model: The model to use.
    :param encodings: The encoded tokens to turn into features.
    :return: The mean of the output for each token.
    """
    encodings = {k: v.to(model.device) for k, v in encodings.items()}
    encoded: BaseModelOutputWithPoolingAndCrossAttentions = model(**encodings)
    out: torch.Tensor = encoded.last_hidden_state.cpu()
    # NOTE: If the dtype is bfloat 16, we convert to float32,
    # because numpy does not suport bfloat16
    # See here: https://github.com/numpy/numpy/issues/19808
    if out.dtype == torch.bfloat16:
        out = out.float()

    # Take the mean by averaging over the attention mask.
    mask = encodings["attention_mask"].cpu().float()
    mask /= mask.sum(1)[:, None]

    result = torch.bmm(mask[:, None, :].float(), out).squeeze(1)

    return result
