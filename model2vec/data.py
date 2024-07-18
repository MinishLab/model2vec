# -*- coding: utf-8 -*-
import logging
from typing import Any, Iterator

from datasets import load_dataset

logger = logging.getLogger(__name__)


def stream_text_from_dataset(
    dataset_name: str, subset_name: str | None = None, text_column_name: str = "text", **kwargs: Any
) -> Iterator[str]:
    """
    Stream text from a huggingface dataset.

    NOTE: kwargs are passed to the `load_dataset` function from the `datasets` library.

    :param dataset_name: The dataset to load using the `load_dataset` function.
    :param subset_name: The name of the subset, e.g., "train".
    :param text_column_name: The column in the dataset containing the text.
    :return: Defines a generator over the text column
    """
    logger.info("Loading dataset")
    dataset = load_dataset(dataset_name, streaming=True, split=subset_name, **kwargs)

    logger.info("Dataset loaded")
    for record in dataset:
        yield record[text_column_name]
