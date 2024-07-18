import json
import logging
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from tokenizers.pre_tokenizers import Whitespace

from model2vec.data import stream_text_from_dataset
from model2vec.logging_config import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    save_folder: Annotated[str, typer.Option(help="The folder to save the word count dictionaries to.")],
    save_steps: Annotated[int, typer.Option(help="The number of steps after which to save the IDF dictionaries.")],
) -> None:
    tokenizer = Whitespace()

    save_folder_path = Path(save_folder)
    if not save_folder_path.exists():
        save_folder_path.mkdir(parents=True)

    word_counts: Counter = Counter()
    doc_counts: Counter = Counter()

    n_toks = 0

    corpus_file = open("corpus.txt", "w")

    for idx, record in enumerate(
        stream_text_from_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", subset_name="train")
    ):
        tokens, _ = zip(*tokenizer.pre_tokenize_str(record))
        s = " ".join(tokens)
        corpus_file.write(f"{s}\n")
        word_counts.update(tokens)
        doc_counts.update(set(tokens))

        if idx > 0 and idx % save_steps == 0:
            save_index = idx // save_steps

            save_path = f"{save_folder}/word_counts_{save_index}.json"
            logger.info(f"Saving word counts to {save_path}")
            with open(save_path, "w") as f:
                json.dump(word_counts, f)
            save_path = f"{save_folder}/doc_counts_{save_index}.json"

            logger.info(f"Saving doc counts to {save_path}")
            with open(save_path, "w") as f:
                json.dump(doc_counts, f)

            n_toks += sum(word_counts.values())
            logger.info(f"Total number of tokens: {n_toks}")
            word_counts = Counter()
            doc_counts = Counter()

    corpus_file.close()


if __name__ == "__main__":
    setup_logging()
    typer.run(main)
