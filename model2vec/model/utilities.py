# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Hashable, Literal, Protocol, cast

import numpy as np
import torch
from reach import Reach
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


PathLike = str | Path

_DEFAULT_BATCH_SIZE = 1024
OutputValue = Literal["sentence_embedding", "token_embeddings"]


class ModulewithWeights(Protocol):
    weight: torch.nn.Parameter


def create_input_embeddings_from_model_name(
    model_name: PathLike, module_path: tuple[str, ...] = ("embeddings", "word_embeddings")
) -> Reach:
    """
    This function creates input embeddings from a model name and a module path.

    :param model_name: The model name to use.
    :param module_path: The module path to use.
    :return: The Reach input embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(model_name)
    tokens = _get_tokens_from_tokenizer(tokenizer)

    module: ModulewithWeights = transformer
    for path in module_path:
        module = getattr(module, path)

    weight: torch.Tensor = module.weight.detach()
    if len(tokens) > weight.shape[0]:
        raise ValueError("Got more tokens than embedding weights. No idea what to do.")
    if weight.shape[0] > len(tokens):
        logger.warning(
            "Num input embeddings and num tokens does not match. Truncating embeddings. Please check if this is correct."
        )
        weight = weight[: len(tokens)]

    name = Path(model_name).stem.replace("_", "-")
    embeddings = Reach(weight.numpy(), list(tokens), name=f"input_{name}")

    return embeddings


def create_output_embeddings_from_model_name(
    model_name: PathLike,
    device: str = "cpu",
    output_value: OutputValue = "token_embeddings",
    include_eos_bos: bool = False,
) -> Reach:
    """
    This function creates output embeddings from a model name.

    It does a forward pass for all tokens in the vocabulary.

    :param model_name: The model name to use.
    :param device: The torch device to use.
    :param output_value: The output value to pass to sentence transformers. If this is 'sentence_embedding', get pooled output, if this is 'token_embedding', get token means.
    :param include_eos_bos: Whether to include the eos and bos tokens in the mean. Only applied if output_value == "token_embeddings".
    :return: The Reach output embeddings.
    """
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokens = _get_tokens_from_tokenizer(tokenizer)

    weight = create_output_embeddings_from_model_name_and_tokens(
        model_name, list(tokens), device, output_value, include_eos_bos
    )
    name = Path(model_name).stem.replace("_", "-")
    embeddings = Reach(weight, list(tokens), name=f"input_{name}")

    return embeddings


def create_output_embeddings_from_model_name_and_reach(
    model_name: PathLike,
    reach: Reach,
    device: str = "cpu",
    output_value: OutputValue = "token_embeddings",
    include_eos_bos: bool = False,
) -> Reach:
    """
    This function creates output embeddings for a bunch of tokens from a model name and reach instance.

    It does a forward pass for all tokens in the reach vocabulary.

    :param model_name: The model name to use.
    :param reach: An existing reach instance.
    :param device: The torch device to use.
    :param output_value: The output value to pass to sentence transformers. If this is 'sentence_embedding', get pooled output, if this is 'token_embedding', get token means.
    :param include_eos_bos: Whether to include the eos and bos tokens in the mean. Only applied if output_value == "token_embeddings".
    :return: The Reach output embeddings.
    """
    tokens = cast(list[str], reach.sorted_items)
    return create_output_embeddings_from_model_name_and_tokens(
        model_name, tokens, device, output_value, include_eos_bos
    )


def create_output_embeddings_from_model_name_and_tokens(
    model_name: PathLike,
    tokens: list[str],
    device: str,
    output_value: Literal["sentence_embedding", "token_embeddings"],
    include_eos_bos: bool,
) -> Reach:
    """
    This function creates output embeddings for a bunch of tokens from a model name

    It does a forward pass for all tokens passed in tokens

    :param model_name: The model name to use.
    :param device: The torch device to use.
    :param output_value: The output value to pass to sentence transformers. If this is 'sentence_embedding', get pooled output, if this is 'token_embedding', get token means.
    :param include_eos_bos: Whether to include the eos and bos tokens in the mean. Only applied if output_value == "token_embeddings".
    :return: The Reach output embeddings.
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
                    logger.warning(f"Got empty token vectors for word {batch[idx]}")
                    mean_vector = np.zeros_like(intermediate_weights[-1])
                else:
                    mean_vector = token_vectors.cpu().numpy().mean(0)
                intermediate_weights.append(mean_vector)
        out_weights = np.stack(intermediate_weights)
    else:
        out_weights = cast(
            np.ndarray,
            embedder.encode(tokens, show_progress_bar=True, output_value=output_value, batch_size=_DEFAULT_BATCH_SIZE),
        )

    match (output_value, include_eos_bos):
        case "token_embeddings", True:
            suffix = "mean"
        case "token_embeddings", False:
            suffix = "mean-no-eos"
        case "sentence_embedding", _:
            suffix = "pooler"

    name = Path(model_name).stem.replace("_", "-")
    embeddings = Reach(out_weights, cast(list[Hashable], tokens), name=f"model2vec_{name}_{suffix}")

    return embeddings


def _get_tokens_from_tokenizer(tokenizer: PreTrainedTokenizerFast) -> tuple[str, ...]:
    """
    Gets all tokens from the vocabulary of a tokenizer in sort order.

    :param tokenizer: The tokenizer from which to get the vocabulary.
    :return: A list of tokens, sorted by vocabulary index.
    """
    tokens: tuple[str, ...]
    tokens, _ = zip(*sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))

    return tokens


def create_tokenizer_from_vocab(vocabulary: dict[str, int], unk_token: str, pad_token: str) -> PreTrainedTokenizerFast:
    """
    Creates a _word_ level tokenizer from a pre-specified vocabulary and an unk token.

    This tokenizer automatically normalizes, and splits using the Whitespace splitter.

    :param vocabulary: The vocabulary mapping from strings to integers.
    :param unk_token: The string representing the unk token.
    :return: A word-level tokenizer.
    """
    if unk_token not in vocabulary:
        raise ValueError(f"unk token: {unk_token} was not in your vocabulary.")

    model = WordLevel(vocab=vocabulary, unk_token=unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Whitespace()

    # NOTE: unk token can be cased, doesn't matter.
    is_cased = any(token != token.lower() for token in vocabulary if token not in {unk_token, pad_token})

    normalization_steps = [NFKC()]
    if not is_cased:
        normalization_steps.append(Lowercase())
    tokenizer.normalizer = Sequence(normalization_steps)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({"unk_token": unk_token, "pad_token": pad_token})

    return tokenizer


def safe_load_reach(path: PathLike) -> Reach:
    """
    Safely load a reach instance from a path.

    NOTE: This tries to load the Reach instance using `load` if the path exists, otherwise,
    it tries to load it as a fast format.

    :param path: The path to load the reach instance from.
    :return: The loaded reach instance.
    """
    if not Path(path).exists():
        return Reach.load_fast_format(path)
    return Reach.load(path)


def add_token_to_reach(embeddings: Reach, token: str, set_as_unk: bool) -> Reach:
    """
    Adds a token to a reach instance if it does not already exist.

    NOTE: this function modifies the input embeddings in place.

    :param embeddings: The embeddings to add the unk token to.
    :param token: The unk token to add.
    :param set_as_unk: Whether to set the token as the unk token.
    :return: The embeddings with the token added.
    """
    if token in embeddings.items:
        logger.info(f"Found {token} in vocabulary. Not adding.")
    else:
        logger.info(f"Adding new token: {token} to vocabulary")
        idx = len(embeddings.items)
        embeddings._items[token] = idx
        embeddings._indices[idx] = token
        embeddings._vectors = np.concatenate([embeddings._vectors, np.zeros((1, embeddings._vectors.shape[1]))])
    if set_as_unk:
        logger.info(f"Setting {token} as unk token.")
        embeddings.unk_index = embeddings.items[token]

    return embeddings
