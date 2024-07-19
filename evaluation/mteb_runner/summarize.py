# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from evaluation.mteb_runner.calculate_baseline import ResultSet, get_results_model_folder


def main() -> dict[str, ResultSet]:
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="The root folder of your results directory.", default="results")
    parser.add_argument("--baseline", type=str, help="The baseline directory.", default=None)
    args = parser.parse_args()

    results = {}
    for model_name_path in Path(args.results_dir).iterdir():
        name = model_name_path.name
        results[name] = get_results_model_folder(model_name_path)

    if args.baseline is not None:
        results["baseline"] = get_results_model_folder(Path(args.baseline))

    return results


if __name__ == "__main__":
    results = main()

    if "baseline" in results:
        baseline = results["baseline"].summarize()
        d = {k: v.summarize() - baseline for k, v in results.items()}
    else:
        d = {k: v.summarize() for k, v in results.items()}

    task_scores = {}
    for task_subset in ["STS", "Classification", "Clustering", "PairClassification", "Reranking", "Summarization"]:
        if "baseline" in results:
            baseline = results["baseline"].summarize(task_subset=task_subset)
            task_scores[task_subset] = pd.DataFrame(
                {k: v.summarize(task_subset=task_subset) - baseline for k, v in results.items()}
            )
        else:
            task_scores[task_subset] = pd.DataFrame(
                {k: v.summarize(task_subset=task_subset) for k, v in results.items()}
            )
