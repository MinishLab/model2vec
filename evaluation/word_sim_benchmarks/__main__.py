# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from argparse import ArgumentParser

from evaluation.mteb_runner.encoders import StaticEmbedder
from evaluation.word_sim_benchmarks.utilities import calculate_spearman_correlation
from model2vec.logging_config import setup_logging
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging 

logger = logging.getLogger(__name__)

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model-name", help="The model to use.", required=True)
    parser.add_argument("--vocabulary", help="The vocabulary to use (optional).", required=False)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    embedder = SentenceTransformer(model_name_or_path=args.model_name, device="cpu")
    embedder = embedder.eval().to(args.device)
    model_name = Path(args.model_name).name.replace("_", "-")
    name = f"sentencetransformer_{model_name}"
    
    if args.suffix:
        name = f"{name}_{args.suffix}"

    # Define the path the JSONL file containing paths and info about the tasks to evaluate
    tasks_file_path = "evaluation/word_sim_benchmarks/tasks.jsonl"
    # Define the path to the vocab file
    vocab_path = "evaluation/word_sim_benchmarks/data/all_vocab.txt"

    # Read the JSONL data from the file into a list of tasks
    with open(tasks_file_path, "r") as file:
        tasks = [json.loads(line) for line in file]

    # Read the vocab file into a list of words
    with open(vocab_path, "r") as file:
        vocab = [line.strip() for line in file]
        
    embeddings_list = embedder.encode(vocab, batch_size=64)  # Adjust batch_size as needed
    embeddings = {word: embedding for word, embedding in zip(vocab, embeddings_list)}

    #embeddings = {word: embedder.encode(word) for word in vocab}
    all_scores = {}

    # Iterate over the tasks and calculate the Spearman correlation
    for task in tasks:
        score = calculate_spearman_correlation(
            file_path=task["file"],
            vectors=embeddings,
            index1=task["index1"],
            index2=task["index2"],
            index_target=task["target"],
        )
        all_scores[task["task"]] = score

    # Calculate the average score
    all_scores["average"] = round(sum(all_scores.values()) / len(all_scores), 3)

    # Create the results directory if it does not exist
    Path(f"results/{name}").mkdir(parents=True, exist_ok=True)
    
    # Save the scores json to a file
    with open(f"results/{name}/word_sim_benchmarks.json", "w") as file:
        json.dump(all_scores, file, indent=4)
        
    logger.info(all_scores)
    logger.info(f"Results saved to results/{name}/word_sim_benchmarks.json")
    


if __name__ == "__main__":
    setup_logging()

    main()
