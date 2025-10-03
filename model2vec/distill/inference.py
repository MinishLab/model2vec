# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import logging
from enum import Enum
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

PathLike = Union[Path, str]
PCADimType = Union[int, None, float, Literal["auto"]]

_DEFAULT_BATCH_SIZE = 256


class PoolingType(str, Enum):
    """
    Pooling strategies for embedding creation.

    - MEAN: masked mean over all tokens (ignores padding).
    - LAST: last non-padding token (often EOS, common in decoder-style models).
    - FIRST: first token hidden state (position 0). In BERT-style encoders,
               this corresponds to the [CLS] token representation.
    - POOLER: use the model's `pooler_output`. In BERT-like models this is
               computed as the hidden state at [CLS], passed through a learned
               dense layer + activation. Not all models provide this.
    """

    MEAN = "mean"
    LAST = "last"
    FIRST = "first"
    POOLER = "pooler"


def create_embeddings(
    model: PreTrainedModel,
    tokenized: list[list[int]],
    device: str,
    pad_token_id: int,
    pooling: PoolingType = PoolingType.MEAN,
) -> np.ndarray:
    """
    Create output embeddings for a bunch of tokens using a pretrained model.

    It does a forward pass for all tokens passed in `tokens`.

    :param model: The model to use.
        This should be a transformers model.
    :param tokenized: All tokenized tokens.
    :param device: The torch device to use.
    :param pad_token_id: The pad token id. Used to pad sequences.
    :param pooling: The pooling strategy to use.
    :return: The output embeddings.
    :raises ValueError: If the pooling strategy is unknown.
    """
    model = model.to(device).eval()  # type: ignore  # Transformers error

    out_weights: np.ndarray
    intermediate_weights: list[np.ndarray] = []

    # Add token_type_ids only if the model supports it
    add_token_type_ids = "token_type_ids" in inspect.getfullargspec(model.forward).args

    lengths = np.asarray([len(sequence) for sequence in tokenized])
    sort_order = np.argsort(lengths)

    sorted_tokenized = [tokenized[i] for i in sort_order]

    pbar = tqdm(total=len(sorted_tokenized), desc="Encoding tokens", unit=" tokens")

    for batch_idx in range(0, len(sorted_tokenized), _DEFAULT_BATCH_SIZE):
        batch_list = sorted_tokenized[batch_idx : batch_idx + _DEFAULT_BATCH_SIZE]
        batch = [torch.tensor(x, dtype=torch.long) for x in batch_list]

        encoded = {}
        encoded["input_ids"] = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)

        # Create attention mask by using the lengths of each sequence
        seq_len = encoded["input_ids"].size(1)
        batch_lengths = torch.tensor([len(x) for x in batch_list], device=encoded["input_ids"].device)
        token_positions = torch.arange(seq_len, device=encoded["input_ids"].device)
        # Mark padding tokens with 0, and non-padding tokens with 1
        attention_mask = token_positions.unsqueeze(0) < batch_lengths.unsqueeze(1)
        encoded["attention_mask"] = attention_mask.to(dtype=torch.long)

        if add_token_type_ids:
            # Add token_type_ids for models that support it
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

        if pooling == PoolingType.MEAN:
            out = _encode_mean_with_model(model, encoded)
        elif pooling == PoolingType.LAST:
            out = _encode_last_with_model(model, encoded)
        elif pooling == PoolingType.FIRST:
            out = _encode_first_with_model(model, encoded)
        elif pooling == PoolingType.POOLER:
            out = _encode_pooler_with_model(model, encoded)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        intermediate_weights.extend(out.numpy())
        pbar.update(len(batch))

    # Sort the output back to the original order
    intermediate_weights = [intermediate_weights[i] for i in np.argsort(sort_order)]
    out_weights = np.stack(intermediate_weights)
    out_weights = np.nan_to_num(out_weights)

    return out_weights


def _encode_with_model(
    model: PreTrainedModel, encodings: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """
    Move inputs to the model device, run a forward pass, and standardize dtypes.

    :param model: The model to use.
    :param encodings: The encoded tokens to turn into features.
    :return: a tuple consisting of:
      - hidden: last_hidden_state
      - pooler: pooler_output if present, else None
      - encodings_on_device: the device-moved encodings (for masks)
    """
    encodings_on_device = {k: v.to(model.device) for k, v in encodings.items()}
    outputs: BaseModelOutputWithPoolingAndCrossAttentions = model(**encodings_on_device)
    hidden: torch.Tensor = outputs.last_hidden_state  # type: ignore  # False positive
    # NOTE: If the dtype is bfloat 16, we convert to float32,
    # because numpy does not suport bfloat16
    # See here: https://github.com/numpy/numpy/issues/19808
    if hidden.dtype == torch.bfloat16:
        hidden = hidden.float()
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None and pooler.dtype == torch.bfloat16:
        pooler = pooler.float()
    return hidden, pooler, encodings_on_device


@torch.inference_mode()
def _encode_mean_with_model(model: PreTrainedModel, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Encode a batch of tokens using mean pooling.

    :param model: The model to use.
    :param encodings: The encoded tokens to turn into features.
    :return: The mean of the output for each token.
    """
    hidden, _, encodings_on_device = _encode_with_model(model, encodings)
    # Take the mean by averaging over the attention mask.
    mask = encodings_on_device["attention_mask"].cpu().float()
    lengths = mask.sum(1, keepdim=True).clamp_min_(1.0)
    mask = mask / lengths
    return torch.bmm(mask.to(hidden.device)[:, None, :], hidden).squeeze(1).cpu()


@torch.inference_mode()
def _encode_last_with_model(model: PreTrainedModel, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Encode a batch of tokens using last token pooling.

    :param model: The model to use.
    :param encodings: The encoded tokens to turn into features.
    :return: The last hidden state for each token.
    """
    hidden, _, encodings_on_device = _encode_with_model(model, encodings)
    mask = encodings_on_device["attention_mask"].bool()
    last_idx = (mask.sum(dim=1) - 1).clamp_min(0).long()
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_indices, last_idx, :].cpu()


@torch.inference_mode()
def _encode_first_with_model(model: PreTrainedModel, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Encode a batch of tokens using first token (CLS) pooling.

    :param model: The model to use.
    :param encodings: The encoded tokens to turn into features.
    :return: The first token representation for each token.
    """
    hidden, _, _ = _encode_with_model(model, encodings)
    return hidden[:, 0, :].cpu()


@torch.inference_mode()
def _encode_pooler_with_model(model: PreTrainedModel, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Encode a batch of tokens using pooler output.

    :param model: The model to use.
    :param encodings: The encoded tokens to turn into features.
    :return: The pooler output for each token.
    :raises ValueError: If the model does not return pooler_output.
    """
    _, pooler, _ = _encode_with_model(model, encodings)
    if pooler is None:
        raise ValueError("POOLER pooling requested, but model did not return pooler_output.")
    return pooler.cpu()


def post_process_embeddings(
    embeddings: np.ndarray, pca_dims: PCADimType, sif_coefficient: float | None = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Post process embeddings by applying PCA and SIF weighting by estimating the frequencies through Zipf's law."""
    if pca_dims is not None:
        if pca_dims == "auto":
            pca_dims = embeddings.shape[1]
        if pca_dims > embeddings.shape[1]:
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of dimensions in the embeddings ({embeddings.shape[1]}). "
                "Applying PCA, but not reducing dimensionality. If this is not desired, set `pca_dims` to None."
            )
            pca_dims = embeddings.shape[1]
        if pca_dims >= embeddings.shape[0]:
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of tokens in the vocabulary ({embeddings.shape[0]}). Not applying PCA."
            )
        elif pca_dims <= embeddings.shape[1]:
            orig_dims = embeddings.shape[1]
            p = PCA(n_components=pca_dims, svd_solver="full")
            embeddings = p.fit_transform(embeddings)
            if embeddings.shape[1] < orig_dims:
                logger.info(
                    f"Reduced dimensionality {orig_dims} -> {embeddings.shape[1]} "
                    f"(explained var ratio: {np.sum(p.explained_variance_ratio_):.3f})."
                )

    if sif_coefficient is not None:
        logger.info("Estimating word frequencies using Zipf's law, and then applying SIF.")
        inv_rank = 1 / (np.arange(2, embeddings.shape[0] + 2))
        proba = inv_rank / np.sum(inv_rank)
        weight = sif_coefficient / (sif_coefficient + proba)
    else:
        weight = np.ones(embeddings.shape[0])

    return embeddings, weight
