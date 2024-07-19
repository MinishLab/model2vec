# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from sentence_transformers import SentenceTransformer

from evaluation.word_sim_benchmarks.utilities import calculate_spearman_correlation, create_vocab_and_tasks_dict
from model2vec.logging_config import setup_logging
from model2vec.model.utilities import create_output_embeddings_from_model_name_and_tokens

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model-name", help="The model to use.", required=True)
    parser.add_argument("--vocabulary-path", help="The vocabulary to use (optional).", required=False)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-type", default="token_embeddings_no_eos")
    args = parser.parse_args()

    embedding_type_used: Literal["token_embeddings", "sentence_embedding"]

    match args.embedding_type:
        case "token_embeddings":
            include_eos_bos = True
            embedding_type_used = "token_embeddings"
        case "sentence_embedding":
            include_eos_bos = True
            embedding_type_used = "sentence_embedding"
        case "token_embeddings_no_eos":
            include_eos_bos = False
            embedding_type_used = "token_embeddings"
        case _:
            raise ValueError(f"Invalid option passed to --embedding-type: {args.embedding_type}")

    logger.info(f"Loading model: {args.model_name}")
    model_name = Path(args.model_name).name.replace("_", "-")

    logger.info(f"Model loaded")

    # Define the path the JSONL file containing paths and info about the tasks to evaluate
    tasks_file_path = "evaluation/word_sim_benchmarks/tasks.jsonl"

    # Read the JSONL data from the file into a list of tasks
    with open(tasks_file_path, "r") as file:
        tasks = [json.loads(line) for line in file]

    # Create the vocabulary and tasks dictionary
    vocab, tasks_dict = create_vocab_and_tasks_dict(tasks)

    # If a vocabulary path is provided, use the intersection
    # of that vocabulary and the task vocabulary instead
    if args.vocabulary_path:
        with open(args.vocabulary_path) as file:
            vocab = list(set(line.strip() for line in file) | set(vocab))

    # Create the embeddings for the vocabulary
    logger.info("Inferring vocabulary.")
    embeddings = create_output_embeddings_from_model_name_and_tokens(
        model_name=args.model_name,
        tokens=vocab,
        device=args.device,
        output_value=embedding_type_used,
        include_eos_bos=include_eos_bos,
    )

    task_scores = {}

    # Iterate over the tasks and calculate the Spearman correlation
    for task_name, task_data in tasks_dict.items():
        score, n_oov_trials = calculate_spearman_correlation(
            task=task_data,
            embeddings=embeddings,
        )
        task_scores[task_name] = score, n_oov_trials

    # Calculate the average score
    all_scores, _ = zip(*task_scores.values())
    average_score = round(sum(all_scores) / len(all_scores))
    task_scores["average"] = average_score

    name = embeddings.name
    if args.suffix:
        name = f"{name}_{args.suffix}"

    # Create the results directory if it does not exist
    Path(f"results/{name}").mkdir(parents=True, exist_ok=True)

    # Save the scores json to a file
    with open(f"results/{name}/word_sim_benchmarks.json", "w") as file:
        json.dump(task_scores, file, indent=4)

    logger.info(task_scores)
    logger.info(f"Results saved to results/{name}/word_sim_benchmarks.json")


if __name__ == "__main__":
    setup_logging()

    main()
