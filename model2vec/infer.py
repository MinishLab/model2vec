# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Hashable, cast

import numpy as np
import typer
from more_itertools import chunked
from reach import Reach
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from typing_extensions import Annotated

from model2vec.data import stream_text_from_dataset
from model2vec.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer()


def _truncate_text(text: str, tokenizer: PreTrainedTokenizerFast) -> str:
    """Truncates text to a model length."""
    max_chars = tokenizer.model_max_length * 10
    text = text[:max_chars]
    # NOTE: we subtract two because of special chars.
    max_length = tokenizer.model_max_length - 2
    return tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"][:max_length])


@app.command()
def main(
    model_name: Annotated[str, typer.Option(help="The model name to use from Hugging Face.")],
    out_folder: Annotated[
        str,
        typer.Option(
            help="The folder to which to write the features. "
            "This script will automatically write a subfolder based on model_name."
        ),
    ] = "features",
    dataset_name: Annotated[str, typer.Option(help="The dataset to use for inference.")] = "allenai/c4",
    save_every_n_batches: Annotated[int, typer.Option(help="The number of batches after which to save.")] = 50,
    device: Annotated[str, typer.Option(help="The device on which to run inference.")] = "cpu",
) -> None:
    """
    This program infers features for the C4 corpus.

    It downloads the model with model_name from hugging face, and infers mean sentence embeddings
    for an unspecified number of items from the C4 corpus. For every n batches, a shard is saved in
    out_folder. These shards are simple reach instances, and numbered sequentially.

    If this process is terminated, it can be resumed by simply using the same model and out_folder.
    """
    embedder = SentenceTransformer(model_name).to(device)

    batch_size = 64

    txts: list[Hashable] = []
    means: list[np.ndarray] = []

    model_name_split = Path(model_name).stem
    path = Path(out_folder) / model_name_split
    path.mkdir(exist_ok=True, parents=True)

    logger.info(path)

    batches = 0
    for idx, x in tqdm(
        enumerate(
            chunked(
                stream_text_from_dataset(
                    dataset_name=dataset_name,
                    subset_name="train",
                    text_column_name="text",
                    name="en",
                ),
                batch_size,
            ),
            start=1,
        )
    ):
        save_index = idx // save_every_n_batches
        next_save_index = save_index + 1

        next_save_path = path / f"{next_save_index}_items.json"
        if next_save_path.exists():
            logger.info(f"{next_save_index} exists, skipping")
            continue

        m = cast(np.ndarray, embedder.encode(x, show_progress_bar=False))
        batches += batch_size

        txts.extend([_truncate_text(x, embedder.tokenizer) for x in x])
        means.extend(m)

        if idx % save_every_n_batches == 0:
            logger.info(f"Processed {batches} items")
            r = Reach(means, txts, name=str(save_index))
            r.save_fast_format(str(path / f"{save_index}"))
            del r

            txts = []
            means = []


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
