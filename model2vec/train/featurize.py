import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from more_itertools import batched
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_SAVE_INTERVAL = 10
logger = logging.getLogger(__name__)


def save_data(means: list[np.ndarray], txts: list[str], base_filepath: str) -> None:
    """
    Save the means and texts to separate files.

    :param means: List of numpy arrays representing the mean embeddings.
    :param txts: List of texts corresponding to the embeddings.
    :param base_filepath: Base path for the output files.
    """
    vectors_filepath = base_filepath + "_vectors.npy"
    items_filepath = base_filepath + "_items.json"

    # Save the embeddings (vectors) to a .npy file
    np.save(vectors_filepath, np.array(means))
    # Save the texts to a JSON file
    with open(items_filepath, "w") as f:
        json.dump({"items": txts}, f)
    logger.info(f"Saved {len(txts)} texts to {items_filepath} and vectors to {vectors_filepath}")


def featurize(
    texts: Iterable[str], model: SentenceTransformer, output_dir: str, max_means: int, batch_size: int
) -> None:
    """
    Featurize text using a sentence transformer.

    :param texts: Iterable of texts to featurize.
    :param model: SentenceTransformer model to use.
    :param output_dir: Directory to save the featurized texts.
    :param max_means: Maximum number of mean embeddings to generate.
    :param batch_size: Batch size to use during encoding. Larger batch sizes may improve speed but require more memory.
    :raises ValueError: If the model does not have a fixed dimension.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_dim = model.get_sentence_embedding_dimension()
    if model_dim is None:
        raise ValueError("Model does not have a fixed dimension")

    txts = []
    means = []
    seen = set()
    total_means = 0

    for index, batch in enumerate(tqdm(batched(texts, batch_size))):
        i = index // _SAVE_INTERVAL
        base_filename = f"featurized_{i}"
        list_batch = [x["text"].strip() for x in batch if x.get("text")]
        if not list_batch:
            continue  # Skip empty batches

        # Encode the batch to get token embeddings
        token_embeddings = model.encode(
            list_batch,
            output_value="token_embeddings",
            convert_to_tensor=True,
        )

        # Tokenize the batch to get input IDs
        tokenized_ids = model.tokenize(list_batch)["input_ids"]

        for tokenized_id, token_embedding in zip(tokenized_ids, token_embeddings):
            # Convert token IDs to tokens (excluding special tokens)
            token_ids = tokenized_id[1:-1]
            # Decode tokens to text
            text = model.tokenizer.decode(tokenized_id, skip_special_tokens=True)
            if text in seen:
                continue
            seen.add(text)
            # Get the corresponding token embeddings (excluding special tokens)
            token_embeds = token_embedding[1:-1]
            # Convert embeddings to NumPy arrays
            token_embeds = token_embeds.detach().cpu().numpy()
            # Compute the mean of the token embeddings
            mean = np.mean(token_embeds, axis=0)
            txts.append(text)
            means.append(mean)
            total_means += 1

            if total_means >= max_means:
                save_data(means, txts, str(out_path / base_filename))
                return

        if index > 0 and (index + 1) % _SAVE_INTERVAL == 0:
            save_data(means, txts, str(out_path / base_filename))
            txts = []
            means = []
            seen = set()
    else:
        if txts and means:
            save_data(means, txts, str(out_path / base_filename))


def main() -> None:
    """Main function to featurize texts using a sentence transformer."""
    parser = argparse.ArgumentParser(description="Featurize texts using a sentence transformer.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="baai/bge-base-en-v1.5",
        help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/c4_bgebase",
        help="Directory to save the featurized texts.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="allenai/c4",
        help="The dataset path or name (e.g. 'allenai/c4').",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="en",
        help="The dataset configuration name (e.g., 'en' for C4).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="The dataset split (e.g., 'train', 'validation').",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode when loading the dataset.",
    )
    parser.add_argument(
        "--max-means",
        type=int,
        default=1000000,
        help="The maximum number of mean embeddings to generate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use for encoding the texts.",
    )

    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)
    dataset = load_dataset(
        args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        streaming=not args.no_streaming,
    )
    featurize(dataset, model, args.output_dir, args.max_means, args.batch_size)


if __name__ == "__main__":
    main()
