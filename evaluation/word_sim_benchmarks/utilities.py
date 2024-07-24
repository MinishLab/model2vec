import logging
from typing import TypedDict

import numpy as np
from reach import normalize
from scipy.stats import spearmanr

from evaluation.utilities import Embedder

logger = logging.getLogger(__name__)


class TaskDescription(TypedDict):
    task: str
    file: str
    index1: int
    index2: int
    target: int


class Task(TypedDict):
    name: str
    words1: list[str]
    words2: list[str]
    targets: list[float]


def create_vocab_and_tasks_dict(
    tasks: list[TaskDescription],
) -> tuple[list[str], dict[str, Task]]:
    """
    Create a vocabulary and a dictionary of task data from a list of tasks.

    :param tasks: A list of tasks.
    :return: A tuple containing the vocabulary and a dictionary of task data.
    """
    vocab = set()
    tasks_dict: dict[str, Task] = {}
    for task in tasks:
        tasks_dict[task["task"]] = {"name": task["task"], "words1": [], "words2": [], "targets": []}
        with open(task["file"], encoding="utf8") as file:
            for line in file:
                # Split the line into words and target value
                line_split = line.strip().split("\t")
                # Lowercase the words and convert the target value to a float
                word1, word2, target = (
                    line_split[task["index1"]].lower(),
                    line_split[task["index2"]].lower(),
                    float(line_split[task["target"]]),
                )
                # Add the words to the vocabulary if they are not already in it
                vocab.update([word1, word2])

                # Add the words and target value to the task dictionary
                tasks_dict[task["task"]]["words1"].append(word1)
                tasks_dict[task["task"]]["words2"].append(word2)
                tasks_dict[task["task"]]["targets"].append(target)

    return list(sorted(vocab)), tasks_dict


def calculate_spearman_correlation(task: Task, embedder: Embedder) -> tuple[float, int]:
    """
    Calculate the Spearman correlation between the similarities of word vectors and a target value.

    :param data: A dictionary containing the words and target values.
    :param embeddings: A dictionary containing the word embeddings.
    :return: The Spearman correlation
    """
    logger.info(f"Doing task {task['name']}")
    similarities = []
    gold_standard = []

    n_oov_trials = 0

    vecs_1 = normalize(embedder.encode(task["words1"]))
    vecs_2 = normalize(embedder.encode(task["words2"]))

    for vec1, vec2, target in zip(vecs_1, vecs_2, task["targets"]):
        # Skip words that are not in the embeddings

        if np.allclose(vec1, 0) or np.allclose(vec2, 0):
            n_oov_trials += 1
            continue

        # Reshape the vectors and calculate the cosine similarity
        similarity_score = normalize(vec1) @ normalize(vec2).T
        similarities.append(similarity_score)
        gold_standard.append(target)

    # Calculate the Spearman correlation
    spearman_score = spearmanr(similarities, gold_standard)[0] * 100

    return spearman_score, n_oov_trials
