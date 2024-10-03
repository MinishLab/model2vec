# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

logger = logging.getLogger(__name__)


PathLike = str | Path

_DEFAULT_BATCH_SIZE = 1024


class ModulewithWeights(Protocol):
    weight: torch.nn.Parameter


def create_output_embeddings_from_model_name_and_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokens: list[str],
    device: str,
) -> tuple[list[str], np.ndarray]:
    """
    Create output embeddings for a bunch of tokens from a model name.

    It does a forward pass for all tokens passed in tokens.

    :param model: The model name to use.
    :param tokenizer: The tokenizer to use.
    :param tokens: The tokens to use.
    :param device: The torch device to use.
    :return: The tokens and output embeddings.
    """
    model = model.to(device)

    out_weights: np.ndarray
    intermediate_weights: list[np.ndarray] = []

    for batch_idx in tqdm(range(0, len(tokens), _DEFAULT_BATCH_SIZE)):
        batch = tokens[batch_idx : batch_idx + _DEFAULT_BATCH_SIZE]
        out = _encode_mean_using_model(model, tokenizer, batch)
        intermediate_weights.append(out.numpy())
    out_weights = np.concatenate(intermediate_weights)

    return tokens, out_weights


@torch.no_grad()
def _encode_mean_using_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, tokens: list[str]) -> torch.Tensor:
    """
    Encode a batch of tokens using a model.

    Note that if a token in the input batch does not have any embeddings, it will be output as a vector of zeros.
    So detection of these is necessary.

    :param model: The model to use.
    :param tokenizer: The tokenizer to use.
    :param tokens: The tokens to encode.
    :return: The mean of the output for each token.
    """
    encodings = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True).to(model.device)
    encoded: BaseModelOutputWithPoolingAndCrossAttentions = model(**encodings)
    out: torch.Tensor = encoded.last_hidden_state.cpu()
    # NOTE: If the dtype is bfloat 16, we convert to float32,
    # because numpy does not suport bfloat16
    # See here: https://github.com/numpy/numpy/issues/19808
    if out.dtype == torch.bfloat16:
        out = out.float()

    mask = encodings["attention_mask"].cpu()
    # NOTE: evil hack. For any batch, there will be a mask vector
    # which has all 1s, because we pad to max_length. argmin returns 0
    # in this case, which is wrong. But because we end up subtracting 1
    # from it, we use -1, which is correct.
    last_nonzero_index = mask.argmin(1) - 1
    # NOTE: do not change the order of these calls. If you do, the hack
    # above will no longer be evil (it will turn good), and will no longer work.
    mask[torch.arange(mask.shape[0]), last_nonzero_index] = 0
    mask[:, 0] = 0

    # We take the mean of embeddings by first summing
    result = torch.bmm(mask[:, None, :].float(), out).squeeze(1)

    # Divide by the number of non-padding tokens, non-cls, etc. tokens.
    divisor = mask.sum(1)
    # Account for the case where divisor is 0.
    divisor[divisor == 0] = 1

    return result / divisor[:, None]


def create_output_embeddings_from_model_name(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> tuple[list[str], np.ndarray]:
    """
    Create output embeddings for a bunch of tokens from a model name.

    It does a forward pass for all ids in the tokenizer.

    :param model: The model name to use.
    :param tokenizer: The tokenizer to use.
    :param device: The torch device to use.
    :return: The tokens and output embeddings.
    """
    model = model.to(device)
    ids = torch.arange(tokenizer.vocab_size)

    # Work-around to get the eos and bos token ids without having to go into tokenizer internals.
    dummy_encoding = tokenizer.encode("A")
    eos_token_id, bos_token_id = dummy_encoding[0], dummy_encoding[-1]

    eos = torch.full([len(ids)], fill_value=eos_token_id)
    bos = torch.full([len(ids)], fill_value=bos_token_id)

    stacked = torch.stack([bos, ids, eos], dim=1)

    intermediate_weights: list[np.ndarray] = []
    for batch_idx in tqdm(range(0, len(stacked), _DEFAULT_BATCH_SIZE)):
        batch = stacked[batch_idx : batch_idx + _DEFAULT_BATCH_SIZE].to(model.device)
        with torch.no_grad():
            # NOTE: we create these masks because nomic embed requires them.
            # Normally, we could set them to None
            token_type_ids = torch.zeros_like(batch)
            attention_mask = torch.ones_like(batch)
            encoded: BaseModelOutputWithPoolingAndCrossAttentions = model(
                input_ids=batch.to(device), attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            out: torch.Tensor = encoded.last_hidden_state
            # NOTE: If the dtype is bfloat 16, we convert to float32,
            # because numpy does not suport bfloat16
            # See here: https://github.com/numpy/numpy/issues/19808
            if out.dtype == torch.bfloat16:
                out = out.float()
        intermediate_weights.append(out[:, 1].cpu().numpy())
    out_weights = np.concatenate(intermediate_weights)

    return tokenizer.convert_ids_to_tokens(ids), out_weights
