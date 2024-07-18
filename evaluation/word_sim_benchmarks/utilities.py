import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


def calculate_spearman_correlation(
    file_path: str, vectors: dict[str, np.ndarray], index1: int, index2: int, index_target: int
):
    """Calculate the Spearman correlation between the similarities of word vectors and a target value.

    :param file_path: The path to the file containing the word pairs and target values.
    :param vectors: The word vectors to use.
    :param index1: The index of the first word in the file.
    :param index2: The index of the second word in the file.
    :param index_target: The index of the target value in the file.
    :return: The Spearman correlation
    """
    similarities = []
    gold_standard = []

    with open(file_path, encoding="utf8") as file:
        for line in file:
            # Split the line into words and target value
            line_split = line.strip().split("\t")
            word1, word2 = line_split[index1].lower(), line_split[index2].lower()

            # Check if the words are in the vectors
            if word1 in vectors and word2 in vectors:
                # Reshape the vectors and calculate the cosine similarity
                v1, v2 = vectors[word1].reshape(1, -1), vectors[word2].reshape(1, -1)
                similarity_score = cosine_similarity(v1, v2)[0][0]
                similarities.append(similarity_score)
                gold_standard.append(float(line_split[index_target]))

    # Calculate the Spearman correlation
    spearman_score = stats.spearmanr(similarities, gold_standard)[0] * 100

    return spearman_score
