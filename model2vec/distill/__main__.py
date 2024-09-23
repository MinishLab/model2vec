import logging
from collections import Counter
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from sklearn.decomposition import PCA
from tokenizers import Tokenizer

from model2vec.distill.inference import (
    create_output_embeddings_from_model_name,
    create_output_embeddings_from_model_name_and_tokens,
)
from model2vec.distill.tokenizer import add_tokens, remove_tokens
from model2vec.model import StaticModel
from model2vec.utils import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    model_name: Annotated[str, typer.Option(help="The model name to initialize the embedder model with.")],
    save_path: Annotated[str, typer.Option(help="The folder to save the model to.")],
    vocabulary_path: Annotated[
        Optional[str],
        typer.Option(
            help="The path to the vocabulary file, which is a .txt file with one word per line. If this is not passed, we use the model's vocabulary."
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="The device to train the model on.")] = "cpu",
) -> None:
    """Creates output embeddings for a sentencetransformer."""
    if vocabulary_path is not None:
        vocabulary = open(vocabulary_path).read().splitlines()
    else:
        vocabulary = None

    model = distill(model_name, vocabulary, device)
    model.save_pretrained(Path(save_path))


def distill(
    model_name: str,
    vocabulary: list[str] | None = None,
    device: str = "cpu",
    pca_dims: int | None = 256,
    apply_zipf: bool = True,
) -> StaticModel:
    """
    Distill down a sentencetransformer to a static model.

    This function creates a set of embeddings from a sentencetransformer. It does this by doing either
    a forward pass for all subword tokens in the tokenizer, or by doing a forward pass for all tokens in a passed vocabulary.

    If you pass through a vocabulary, we create a custom word tokenizer for that vocabulary.
    If you don't pass a vocabulary, we use the model's tokenizer directly.

    :param model_name: The model name to use. Any sentencetransformer compatible model works.
    :param vocabulary: The vocabulary to use. If this is None, we use the model's vocabulary.
    :param device: The device to use.
    :param pca_dims: The number of components to use for PCA. If this is None, we don't apply PCA.
    :param apply_zipf: Whether to apply Zipf weighting to the embeddings.
    :raises: ValueError if the PCA dimension is larger than the number of dimensions in the embeddings.
    :raises: ValueError if the vocabulary contains duplicate tokens.
    :return: A StaticModdel

    """
    tokenizer: Tokenizer = Tokenizer.from_pretrained(model_name)
    tokens, embeddings = create_output_embeddings_from_model_name(model_name, device=device)
    tokenizer_name = model_name

    wrong_tokens = [x for x in tokens if x.startswith("[unused")]
    vocab = tokenizer.get_vocab()
    wrong_token_ids = [vocab[token] for token in wrong_tokens]
    tokenizer = remove_tokens(tokenizer, wrong_tokens)
    embeddings = np.delete(embeddings, wrong_token_ids, axis=0)
    logger.info("Removed unused tokens from the tokenizer and embeddings.")

    if vocabulary is not None:
        vocabulary_counts = Counter(vocabulary)
        duplicates = [k for k, v in vocabulary_counts.items() if v > 1]
        if duplicates:
            duplicate_str = ", ".join(duplicates)
            raise ValueError(f"Vocabulary contains duplicate tokens: {duplicate_str}")

        subword_tokens_set = set(tokens)
        # NOTE: we do a list comprehension to keep sort order.
        n_tokens_before = len(vocabulary)
        new_tokens = [token for token in vocabulary if token not in subword_tokens_set]
        n_tokens_after = len(vocabulary)
        logger.info(
            f"Adding {len(new_tokens)} tokens to the vocabulary. Removed {n_tokens_before - n_tokens_after} tokens."
        )
        token_embeddings = create_output_embeddings_from_model_name_and_tokens(
            model_name=model_name,
            tokens=new_tokens,
            device=device,
            output_value="token_embeddings",
            include_eos_bos=False,
        )

        tokenizer = add_tokens(tokenizer, new_tokens)
        embeddings = np.concatenate([embeddings, token_embeddings], axis=0)
        tokens += new_tokens

    if pca_dims is not None:
        if pca_dims >= embeddings.shape[1]:
            raise ValueError(
                f"PCA dimension ({pca_dims}) is larger than the number of dimensions in the embeddings ({embeddings.shape[1]})"
            )
        if pca_dims >= len(tokens):
            logger.warning(
                f"PCA dimension ({pca_dims}) is larger than the number of tokens in the vocabulary ({len(tokens)}). Not applying PCA."
            )
        elif pca_dims < embeddings.shape[1]:
            logger.info(f"Applying PCA with n_components {pca_dims}")

            p = PCA(n_components=pca_dims, whiten=False)
            embeddings = p.fit_transform(embeddings)

    if apply_zipf:
        logger.info("Applying Zipf weighting")
        w = np.log(np.arange(1, len(embeddings) + 1))
        embeddings *= w[:, None]

    config = {"tokenizer_name": tokenizer_name, "apply_pca": pca_dims, "apply_zipf": apply_zipf}

    return StaticModel(embeddings, tokenizer, config)


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
