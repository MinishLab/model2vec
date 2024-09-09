import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from transformers import AutoTokenizer

from korok.model2vec.infer import (
    create_output_embeddings_from_model_name,
    create_output_embeddings_from_model_name_and_tokens,
)
from korok.model2vec.tokenizer import create_tokenizer_from_vocab
from korok.utils import save_pretrained, setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    model_name: Annotated[str, typer.Option(help="The model name to initialize the embedder model with.")],
    save_path: Annotated[str, typer.Option(help="The folder to save the model to.")],
    vocabulary_path: Annotated[
        Optional[str],
        typer.Option(help="The path to the vocabulary file. If this is passed, the features_path option is ignored."),
    ] = None,
    device: Annotated[str, typer.Option(help="The device to train the model on.")] = "cpu",
) -> None:
    """
    Create output embeddings for a bunch of tokens from a model name and reach instance.

    It does a forward pass for all tokens in the reach vocabulary.
    """
    if vocabulary_path is None:
        tokens, embeddings = create_output_embeddings_from_model_name(model_name, device=device)
        vocabulary_path_name = ""
        tokenizer_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokens = open(vocabulary_path).read().splitlines()
        if "[PAD]" not in tokens:
            tokens = ["[PAD]"] + tokens
        if "[UNK]" not in tokens:
            tokens = ["[UNK]"] + tokens
        tokens, embeddings = create_output_embeddings_from_model_name_and_tokens(
            model_name, tokens, device=device, output_value="token_embeddings", include_eos_bos=False
        )
        vocabulary_path_name = Path(vocabulary_path).name
        tokenizer_name = "word_level"
        tokenizer = create_tokenizer_from_vocab(tokens, unk_token="[UNK]", pad_token="[PAD]")

    save_pretrained(
        Path(save_path), embeddings, tokenizer, {"tokenizer_name": tokenizer_name, "vocabulary": vocabulary_path_name}
    )


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
