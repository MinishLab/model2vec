import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from datasets import Dataset, DatasetInfo

from korok.model2vec.utils import create_output_embeddings_from_model_name_and_tokens
from korok.utils import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    model_name: Annotated[str, typer.Option(help="The model name to initialize the embedder model with.")],
    save_path: Annotated[str, typer.Option(help="The folder to save the model to.")],
    vocabulary_path: Annotated[
        str,
        typer.Option(help="The path to the vocabulary file. If this is passed, the features_path option is ignored."),
    ],
    device: Annotated[str, typer.Option(help="The device to train the model on.")] = "cpu",
) -> None:
    """
    Create output embeddings for a bunch of tokens from a model name and reach instance.

    It does a forward pass for all tokens in the reach vocabulary.
    """
    tokens = open(vocabulary_path).read().splitlines()
    tokens, vectors = create_output_embeddings_from_model_name_and_tokens(
        model_name, tokens, device=device, output_value="token_embeddings", include_eos_bos=False
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    metadata_dict = {
        "description": "Model2Vec embeddings.",
        "vocabulary_path": Path(vocabulary_path).name,
        "model_name": model_name,
        "tokenizer": "word",
    }
    info = DatasetInfo(description=json.dumps(metadata_dict))
    dataset = Dataset.from_dict({"tokens": tokens, "vectors": vectors}, info=info).with_format("numpy")
    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
