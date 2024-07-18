import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


def create_vocab_and_tasks_dict(
    tasks: list[dict[str, str | int]],
) -> tuple[list[str], dict[str, dict[str, list[str | float]]]]:
    """
    Create a vocabulary and a dictionary of task data from a list of tasks.

    :param tasks: A list of tasks.
    :return: A tuple containing the vocabulary and a dictionary of task data.
    """
    vocab = []
    tasks_dict: dict[str, dict[str, list[str | float]]] = {}
    for task in tasks: 
        tasks_dict[str(task["task"])] = {"words1": [], "words2": [], "targets": []}
        with open(task["file"], encoding="utf8") as file:
            for line in file:
                # Split the line into words and target value
                line_split = line.strip().split("\t")
                # Lowercase the words and convert the target value to a float
                word1, word2, target = (
                    line_split[int(task["index1"])].lower(),
                    line_split[int(task["index2"])].lower(),
                    float(line_split[int(task["target"])]),
                )
                # Add the words to the vocabulary if they are not already in it
                if word1 not in vocab:
                    vocab.append(word1)
                if word2 not in vocab:
                    vocab.append(word2)
                    
                # Add the words and target value to the task dictionary
                tasks_dict[str(task["task"])]["words1"].append(word1)
                tasks_dict[str(task["task"])]["words2"].append(word2)
                tasks_dict[str(task["task"])]["targets"].append(target)

    return vocab, tasks_dict


def calculate_spearman_correlation(data: dict[str, list[str | float]], embeddings: dict[str, np.ndarray]):
    """
    Calculate the Spearman correlation between the similarities of word vectors and a target value.

    :param data: A dictionary containing the words and target values.
    :param embeddings: A dictionary containing the word embeddings.
    :return: The Spearman correlation
    """
    similarities = []
    gold_standard = []

    for word1, word2, target in zip(data["words1"], data["words2"], data["targets"]):
        # Reshape the vectors and calculate the cosine similarity
        v1, v2 = embeddings[str(word1)].reshape(1, -1), embeddings[str(word2)].reshape(1, -1)
        similarity_score = cosine_similarity(v1, v2)[0][0]
        similarities.append(similarity_score)
        gold_standard.append(target)

    # Calculate the Spearman correlation
    spearman_score = stats.spearmanr(similarities, gold_standard)[0] * 100

    return spearman_score
