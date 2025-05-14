# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Literal, Protocol, Union

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

logger = logging.getLogger(__name__)


PathLike = Union[Path, str]
PCADimType = Union[int, None, float, Literal["auto"]]


_DEFAULT_BATCH_SIZE = 256


class ModulewithWeights(Protocol):
    weight: torch.nn.Parameter


def create_embeddings(
    model: PreTrainedModel,
    tokenized: list[list[int]],
    device: str,
    pad_token_id: int,
) -> np.ndarray:
    """
    Create output embeddings for a bunch of tokens using a pretrained model.

    It does a forward pass for all tokens passed in `tokens`.

    :param model: The model to use.
        This should be a transformers model.
    :param tokenized: All tokenized tokens.
    :param device: The torch device to use.
    :param pad_token_id: The pad token id. Used to pad sequences.
    :return: The output embeddings.
    """
    model = model.to(device)

    out_weights: np.ndarray
    intermediate_weights: list[np.ndarray] = []

    # Add token_type_ids only if the model supports it
    add_token_type_ids = "token_type_ids" in inspect.getfullargspec(model.forward).args

    lengths = np.asarray([len(sequence) for sequence in tokenized])
    sort_order = np.argsort(lengths)

    sorted_tokenized = [tokenized[i] for i in sort_order]

    pbar = tqdm(total=len(sorted_tokenized), desc="Encoding tokens", unit=" tokens")

    for batch_idx in range(0, len(sorted_tokenized), _DEFAULT_BATCH_SIZE):
        batch = [torch.Tensor(x).long() for x in sorted_tokenized[batch_idx : batch_idx + _DEFAULT_BATCH_SIZE]]

        encoded = {}
        encoded["input_ids"] = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
        encoded["attention_mask"] = encoded["input_ids"] != pad_token_id

        if add_token_type_ids:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

        out = _encode_mean_using_model(model, encoded)
        intermediate_weights.extend(out.numpy())
        pbar.update(len(batch))

    # Sort the output back to the original order
    intermediate_weights = [intermediate_weights[i] for i in np.argsort(sort_order)]
    out_weights = np.stack(intermediate_weights)

    out_weights = np.nan_to_num(out_weights)

    return out_weights


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


def post_process_embeddings(
    embeddings: np.ndarray, pca_dims: PCADimType, sif_coefficient: float | None = 1e-4
) -> np.ndarray:
    """Post process embeddings by applying PCA and SIF weighting by estimating the frequencies through Zipf's law."""
    if pca_dims is not None:
        if pca_dims == "auto":
            pca_dims = embeddings.shape[1]
        if pca_dims > embeddings.shape[1]:
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of dimensions in the embeddings ({embeddings.shape[1]}). "
                "Applying PCA, but not reducing dimensionality. Is this is not desired, please set `pca_dims` to None. "
                "Applying PCA will probably improve performance, so consider just leaving it."
            )
            pca_dims = embeddings.shape[1]
        if pca_dims >= embeddings.shape[0]:
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of tokens in the vocabulary ({embeddings.shape[0]}). Not applying PCA."
            )
        elif pca_dims <= embeddings.shape[1]:
            if isinstance(pca_dims, float):
                logger.info(f"Applying PCA with {pca_dims} explained variance.")
            else:
                logger.info(f"Applying PCA with n_components {pca_dims}")

            orig_dims = embeddings.shape[1]
            p = PCA(n_components=pca_dims, svd_solver="full")
            embeddings = p.fit_transform(embeddings)

            if embeddings.shape[1] < orig_dims:
                explained_variance_ratio = np.sum(p.explained_variance_ratio_)
                explained_variance = np.sum(p.explained_variance_)
                logger.info(f"Reduced dimensionality from {orig_dims} to {embeddings.shape[1]}.")
                logger.info(f"Explained variance ratio: {explained_variance_ratio:.3f}.")
                logger.info(f"Explained variance: {explained_variance:.3f}.")

    if sif_coefficient is not None:
        logger.info("Estimating word frequencies using Zipf's law, and then applying SIF.")
        inv_rank = 1 / (np.arange(2, embeddings.shape[0] + 2))
        proba = inv_rank / np.sum(inv_rank)
        embeddings *= (sif_coefficient / (sif_coefficient + proba))[:, None]

    return embeddings
