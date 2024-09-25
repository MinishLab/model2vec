import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from model2vec.distill.distillation import distill
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
    pca_dims: Annotated[
        int | None, typer.Option(help="The PCA dimensionality to use. If this is None, no PCA is applied.")
    ] = 256,
    apply_zipf: Annotated[bool, typer.Option(help="Whether to apply Zipf weighting.")] = True,
    use_subword: Annotated[
        bool, typer.Option(help="Whether to use subword tokenization. If this is False, you must pass a vocabulary.")
    ] = True,
) -> None:
    """Creates output embeddings for a sentencetransformer."""
    if vocabulary_path is not None:
        vocabulary = open(vocabulary_path).read().splitlines()
    else:
        vocabulary = None

    model = distill(
        model_name=model_name,
        vocabulary=vocabulary,
        device=device,
        pca_dims=pca_dims,
        apply_zipf=apply_zipf,
        use_subword=use_subword,
    )
    model.save_pretrained(Path(save_path))


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
