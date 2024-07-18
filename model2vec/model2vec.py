import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from tokenlearn.logging_config import setup_logging
from tokenlearn.model.utilities import (
    create_output_embeddings_from_model_name_and_reach,
    create_output_embeddings_from_model_name_and_tokens,
    safe_load_reach,
)

logger = logging.getLogger(__name__)


app = typer.Typer()


@app.command()
def main(
    model_name: Annotated[str, typer.Option(help="The model name to initialize the embedder model with.")],
    save_path: Annotated[str, typer.Option(help="The folder to save the model to.")],
    features_path: Annotated[Optional[str], typer.Option(help="The path to a reach instance.")] = None,
    vocabulary_path: Annotated[
        Optional[str],
        typer.Option(help="The path to the vocabulary file. If this is passed, the features_path option is ignored."),
    ] = None,
    device: Annotated[str, typer.Option(help="The device to train the model on.")] = "cpu",
) -> None:
    """
    This script creates output embeddings for a bunch of tokens from a model name and reach instance.

    It does a forward pass for all tokens in the reach vocabulary.
    """
    if features_path and vocabulary_path:
        raise ValueError("You can only pass one of features_path and vocabulary_path.")

    if features_path:
        reach = safe_load_reach(features_path)
        embeddings = create_output_embeddings_from_model_name_and_reach(
            model_name, reach, device=device, output_value="token_embeddings", include_eos_bos=False
        )
    elif vocabulary_path:
        tokens = open(vocabulary_path).read().splitlines()
        embeddings = create_output_embeddings_from_model_name_and_tokens(
            model_name, tokens, device=device, output_value="token_embeddings", include_eos_bos=False
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    embeddings.save_fast_format(save_path)


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
