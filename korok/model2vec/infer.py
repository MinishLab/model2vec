# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

logger = logging.getLogger(__name__)


PathLike = str | Path

_DEFAULT_BATCH_SIZE = 1024

OutputValue = Literal["sentence_embedding", "token_embeddings"]


class ModulewithWeights(Protocol):
    weight: torch.nn.Parameter


def create_output_embeddings_from_model_name_and_tokens(
    model_name: PathLike,
    tokens: list[str],
    device: str,
    output_value: Literal["sentence_embedding", "token_embeddings"],
    include_eos_bos: bool,
) -> tuple[list[str], np.ndarray]:
    """
    Create output embeddings for a bunch of tokens from a model name.

    It does a forward pass for all tokens passed in tokens.

    :param model_name: The model name to use.
    :param tokens: The tokens to use.
    :param device: The torch device to use.
    :param output_value: The output value to pass to sentence transformers. If this is 'sentence_embedding', get pooled output, if this is 'token_embedding', get token means.
    :param include_eos_bos: Whether to include the eos and bos tokens in the mean. Only applied if output_value == "token_embeddings".
    :return: The tokens and output emnbeddings.
    """
    embedder = SentenceTransformer(str(model_name), device=device)
    out_weights: np.ndarray
    if output_value == "token_embeddings":
        intermediate_weights: list[np.ndarray] = []
        # NOTE: because tokens might be really long, and we want to take the mean anyway, we need to batch.
        # otherwise we could go OOM.
        for batch_idx in tqdm(range(0, len(tokens), _DEFAULT_BATCH_SIZE)):
            batch = tokens[batch_idx : batch_idx + _DEFAULT_BATCH_SIZE]
            out: list[torch.Tensor] = cast(
                list[torch.Tensor], embedder.encode(batch, show_progress_bar=False, output_value=output_value)
            )
            for idx, token_vectors in enumerate(out):
                if not include_eos_bos:
                    # NOTE: remove BOS/EOS
                    token_vectors = token_vectors[1:-1]
                if len(token_vectors) == 0:
                    str_repr = batch[idx]
                    bytes_repr = str_repr.encode("utf-8")
                    logger.warning(f"Got empty token vectors for word `{str_repr}` with bytes `{bytes_repr!r}`")
                    mean_vector = np.zeros_like(intermediate_weights[-1])
                else:
                    mean_vector = cast(np.ndarray, token_vectors.cpu().numpy()).mean(0)
                intermediate_weights.append(mean_vector)
        out_weights = np.stack(intermediate_weights)
    else:
        out_weights = cast(
            np.ndarray,
            embedder.encode(tokens, show_progress_bar=True, output_value=output_value, batch_size=_DEFAULT_BATCH_SIZE),
        )

    return tokens, out_weights


def create_output_embeddings_from_model_name(
    model_name: PathLike,
    device: str,
) -> tuple[list[str], np.ndarray]:
    """
    Create output embeddings for a bunch of tokens from a model name.

    It does a forward pass for all ids in the tokenizer.

    :param model_name: The model name to use.
    :param device: The torch device to use.
    :return: The tokens and output emnbeddings.
    """
    model: PreTrainedModel = AutoModel.from_pretrained(model_name).to(device)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    ids = torch.arange(tokenizer.vocab_size)

    # Work-around
    dummy_encoding = tokenizer.encode("A")
    eos_token_id, bos_token_id = dummy_encoding[0], dummy_encoding[-1]

    eos = torch.full([len(ids)], fill_value=eos_token_id)
    bos = torch.full([len(ids)], fill_value=bos_token_id)

    stacked = torch.stack([bos, ids, eos], dim=1)

    intermediate_weights: list[np.ndarray] = []
    for batch_idx in tqdm(range(0, len(stacked), _DEFAULT_BATCH_SIZE)):
        batch = stacked[batch_idx : batch_idx + _DEFAULT_BATCH_SIZE]
        with torch.no_grad():
            encoded: BaseModelOutputWithPoolingAndCrossAttentions = model(input_ids=batch.to(device))
            out: torch.Tensor = encoded.last_hidden_state
        intermediate_weights.append(out[:, 1].cpu().numpy())
    out_weights = np.concatenate(intermediate_weights)

    return tokenizer.convert_ids_to_tokens(ids), out_weights
