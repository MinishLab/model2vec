# -*- coding: utf-8 -*-
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

from evaluation.mteb_runner.encoders import StaticEmbedder
from model2vec.logging_config import setup_logging

# NOTE: we leave out "Retrieval" because it is too expensive to run.
ALL_TASKS_TYPES = ("Classification", "Clustering", "PairClassification", "Reranking", "STS", "Summarization")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model_path", help="The model to use.", required=False)
    parser.add_argument("--task_types", nargs="+", default=ALL_TASKS_TYPES)
    parser.add_argument("--torch_model", required=False)
    parser.add_argument("--input", action="store_true")
    parser.add_argument("--word-level", action="store_true")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    embedder: StaticEmbedder | SentenceTransformer

    if args.input and args.word_level:
        raise ValueError("Both input and word level were passed.")

    if not (args.model_path or args.torch_model):
        raise ValueError("Either model path or torch model should be passed.")
    if args.model_path:
        if args.input:
            embedder = StaticEmbedder.from_model(args.model_path)
            name = embedder.name
        elif args.word_level:
            embedder = StaticEmbedder.from_vectors(args.model_path)
            name = embedder.name
        else:
            # Always load on CPU
            embedder = SentenceTransformer(model_name_or_path=args.model_path, device="cpu")
            embedder = embedder.eval().to(args.device)
            model_name = Path(args.model_path).name.replace("_", "-")
            name = f"sentencetransformer_{model_name}"

    if args.suffix:
        name = f"{name}_{args.suffix}"

    task_names = [task for task in mteb.MTEB_MAIN_EN.tasks if mteb.get_task(task).metadata.type in args.task_types]

    evaluation = mteb.MTEB(tasks=task_names, task_langs=["en"])
    evaluation.run(embedder, eval_splits=["test"], output_folder=f"results/{name}")


if __name__ == "__main__":
    setup_logging()

    main()
