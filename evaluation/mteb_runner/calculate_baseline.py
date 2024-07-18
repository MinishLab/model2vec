# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.mteb_runner.utilities import ResultSet, get_results_from_hub, get_results_model_folder


def main() -> dict[str, ResultSet]:
    parser = ArgumentParser()
    parser.add_argument("--models", nargs="+", help="The model names on the huggingface model hub.", required=True)
    parser.add_argument("--baseline", type=str, help="The baseline directory.", default=None)
    args = parser.parse_args()

    results = {}
    if args.baseline is not None:
        results["baseline"] = get_results_model_folder(Path(args.baseline))

    for model_name in args.models:
        result = get_results_from_hub(model_name)
        if result is None:
            continue
        results[model_name] = result

    return results


if __name__ == "__main__":
    results = main()

    if "baseline" in results:
        baseline = results["baseline"].summarize()
        d = {k: v.summarize() - baseline for k, v in results.items() if k != "baseline"}
    else:
        d = {k: v.summarize() for k, v in results.items()}

    task_scores = {}
    for task_subset in ("Classification", "Clustering", "PairClassification", "Reranking", "STS", "Summarization"):
        if "baseline" in results:
            baseline = results["baseline"].summarize(task_subset=task_subset)
            task_scores[task_subset] = pd.DataFrame(
                {k: v.summarize(task_subset=task_subset) - baseline for k, v in results.items() if k != "baseline"}
            )
        else:
            task_scores[task_subset] = pd.DataFrame(
                {k: v.summarize(task_subset=task_subset) for k, v in results.items()}
            )
