import logging

import numpy as np
from sklearn.decomposition import PCA
from tokenizers import Tokenizer

from model2vec.distill.inference import (
    create_output_embeddings_from_model_name,
    create_output_embeddings_from_model_name_and_tokens,
)
from model2vec.distill.tokenizer import add_tokens, preprocess_vocabulary, remove_tokens
from model2vec.model import StaticModel

logger = logging.getLogger(__name__)


def distill(
    model_name: str,
    vocabulary: list[str] | None = None,
    device: str = "cpu",
    pca_dims: int | None = 256,
    apply_zipf: bool = True,
    use_subword: bool = True,
) -> StaticModel:
    """
    Distill a staticmodel from a sentence transformer.

    This function creates a set of embeddings from a sentence transformer. It does this by doing either
    a forward pass for all subword tokens in the tokenizer, or by doing a forward pass for all tokens in a passed vocabulary.

    If you pass through a vocabulary, we create a custom word tokenizer for that vocabulary.
    If you don't pass a vocabulary, we use the model's tokenizer directly.

    :param model_name: The model name to use. Any sentencetransformer compatible model works.
    :param vocabulary: The vocabulary to use. If this is None, we use the model's vocabulary.
    :param device: The device to use.
    :param pca_dims: The number of components to use for PCA. If this is None, we don't apply PCA.
    :param apply_zipf: Whether to apply Zipf weighting to the embeddings.
    :param use_subword: Whether to keep subword tokens in the vocabulary. If this is False, you must pass a vocabulary.
    :raises: ValueError if the PCA dimension is larger than the number of dimensions in the embeddings.
    :raises: ValueError if the vocabulary contains duplicate tokens.
    :return: A StaticModdel

    """
    if not use_subword and vocabulary is None:
        raise ValueError("You must pass a vocabulary if you don't use subword tokens.")

    original_tokenizer: Tokenizer = Tokenizer.from_pretrained(model_name)
    tokens: list[str] = []
    if use_subword:
        tokens, embeddings = create_output_embeddings_from_model_name(model_name, device=device)

        wrong_tokens = [x for x in tokens if x.startswith("[unused")]
        vocab = original_tokenizer.get_vocab()
        wrong_token_ids = [vocab[token] for token in wrong_tokens]
        new_tokenizer = remove_tokens(original_tokenizer, wrong_tokens)
        embeddings = np.delete(embeddings, wrong_token_ids, axis=0)
        logger.info(f"Removed {len(wrong_tokens)} unused tokens from the tokenizer and embeddings.")
    else:
        unk_token = original_tokenizer.model.unk_token
        new_tokenizer = remove_tokens(original_tokenizer, list(set(original_tokenizer.get_vocab()) - {unk_token}))
        # We need to set embeddings to None because we don't know the dimensions of the embeddings yet.
        embeddings = None

    if vocabulary is not None:
        preprocessed_vocabulary = preprocess_vocabulary(original_tokenizer, vocabulary)
        n_tokens_before = len(preprocessed_vocabulary)
        cleaned_vocabulary = _clean_vocabulary(preprocessed_vocabulary, tokens)
        n_tokens_after = len(cleaned_vocabulary)
        logger.info(
            f"Adding {n_tokens_after} tokens to the vocabulary. Removed {n_tokens_before - n_tokens_after} tokens during preprocessing."
        )
        if cleaned_vocabulary:
            _, token_embeddings = create_output_embeddings_from_model_name_and_tokens(
                model_name=model_name,
                tokens=cleaned_vocabulary,
                device=device,
                output_value="token_embeddings",
                include_eos_bos=False,
            )

            if embeddings is None:
                embeddings = np.zeros((new_tokenizer.get_vocab_size(), token_embeddings.shape[1]))
            embeddings = np.concatenate([embeddings, token_embeddings], axis=0)
            new_tokenizer = add_tokens(new_tokenizer, cleaned_vocabulary)
        else:
            logger.warning("Didn't create any token embeddings as all tokens were duplicates or empty.")

    embeddings = _post_process_embeddings(np.asarray(embeddings), pca_dims, apply_zipf)

    config = {"tokenizer_name": model_name, "apply_pca": pca_dims, "apply_zipf": apply_zipf}

    return StaticModel(embeddings, new_tokenizer, config)


def _post_process_embeddings(embeddings: np.ndarray, pca_dims: int | None, apply_zipf: bool) -> np.ndarray:
    """Post process embeddings by applying PCA and Zipf weighting."""
    if pca_dims is not None:
        if pca_dims >= embeddings.shape[1]:
            raise ValueError(
                f"PCA dimension ({pca_dims}) is larger than the number of dimensions in the embeddings ({embeddings.shape[1]})"
            )
        if pca_dims >= embeddings.shape[0]:
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of tokens in the vocabulary ({embeddings.shape[0]}). Not applying PCA."
            )
        elif pca_dims < embeddings.shape[1]:
            logger.info(f"Applying PCA with n_components {pca_dims}")

            p = PCA(n_components=pca_dims, whiten=False)
            embeddings = p.fit_transform(embeddings)

    if apply_zipf:
        logger.info("Applying Zipf weighting")
        embeddings *= np.log(1 + np.arange(embeddings.shape[0]))[:, None]

    return embeddings


def _clean_vocabulary(preprocessed_vocabulary: list[str], added_tokens: list[str]) -> list[str]:
    """Cleans a vocabulary by removing duplicates and tokens that were already in the vocabulary."""
    added_tokens_set = set(added_tokens)
    seen_tokens = set()
    cleaned_vocabulary = []
    n_empty = 0
    n_duplicates = 0
    for token in preprocessed_vocabulary:
        if not token:
            n_empty += 1
            continue
        if token in seen_tokens or token in added_tokens_set:
            n_duplicates += 1
            continue
        seen_tokens.add(token)
        cleaned_vocabulary.append(token)

    if n_duplicates:
        logger.warning(f"Removed {n_duplicates} duplicate tokens.")
    if n_empty:
        logger.warning(f"Removed {n_empty} empty tokens.")

    return cleaned_vocabulary