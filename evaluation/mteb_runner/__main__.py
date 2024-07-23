# -*- coding: utf-8 -*-
from __future__ import annotations

from argparse import ArgumentParser

import mteb

from evaluation.utilities import get_default_argparser, load_embedder
from model2vec.logging_config import setup_logging

# NOTE: we leave out "Retrieval" because it is too expensive to run.
ALL_TASKS_TYPES = ("Classification", "Clustering", "PairClassification", "Reranking", "STS", "Summarization")


def main() -> None:
    parser = get_default_argparser()
    parser.add_argument("--task-types", nargs="+", default=ALL_TASKS_TYPES)
    args = parser.parse_args()

    embedder, name = load_embedder(args.model_path, args.input, args.word_level, args.device)

    if args.suffix:
        name = f"{name}_{args.suffix}"

    task_names = [task for task in mteb.MTEB_MAIN_EN.tasks if mteb.get_task(task).metadata.type in args.task_types]

    evaluation = mteb.MTEB(tasks=task_names, task_langs=["en"])
    evaluation.run(embedder, eval_splits=["test"], output_folder=f"results/{name}")


if __name__ == "__main__":
    setup_logging()

    main()
