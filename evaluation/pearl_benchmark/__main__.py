import json
import logging
from pathlib import Path
from typing import Literal, cast

import numpy as np
from autofj.datasets import load_data
from datasets import Dataset, load_dataset
from reach import Reach, normalize
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from evaluation.pearl_benchmark.probing import run_probing_model
from evaluation.utilities import Embedder, get_default_argparser, load_embedder
from model2vec.logging_config import setup_logging

logger = logging.getLogger(__name__)


def eval_bird(model: Embedder, dataset: Dataset):
    input1 = normalize(model.encode(dataset["term1"]))
    input2 = normalize(model.encode(dataset["term2"]))

    sim = (input1 * input2).sum(1)
    sim = (sim + 1) / 2.0
    cor, _ = pearsonr(sim, dataset["relatedness score"])

    return cor


def eval_turney(model: Embedder, dataset: Dataset):
    data_list = []
    for row in dataset:
        data_list.append(
            list(
                (
                    row["query"],
                    row["label"],
                    row["candidate_1"],
                    row["candidate_2"],
                    row["candidate_3"],
                    row["candidate_4"],
                )
            )
        )

    num_correct = 0
    for components in data_list:
        emb = cast(np.ndarray, model.encode(components))
        query = emb[0, :]
        matrix = emb[1:, :]
        scores = np.dot(matrix, query)
        chosen = np.argmax(scores)

        if chosen == 0:
            num_correct += 1
    accuracy = num_correct / len(data_list)

    return accuracy


def eval_ppdb(model: Embedder, dataset: Dataset):
    phrase1_emb = model.encode(dataset["phrase_1"])
    phrase2_emb = model.encode(dataset["phrase_2"])
    label_list = [1 if e == "pos" else 0 for e in dataset["label"]]

    score = run_probing_model(np.concatenate([phrase1_emb, phrase2_emb], axis=1), label_list)

    return score


def eval_clustering(model: Embedder, dataset: Dataset, name: Literal["conll", "bc5cdr"]):
    label_dict = dict()
    match name:
        case "conll":
            label_dict = {"PER": 0, "LOC": 1, "ORG": 2}
        case "bc5cdr":
            label_dict = {"Chemical": 0, "Disease": 1}
        case _:
            raise ValueError(f"Invalid dataset name: {name}")

    num_class = len(label_dict)

    phrases, labels = [], []
    for row in dataset:
        phrases.append(row["entity"] or "NA")
        labels.append(row["label"])

    phrase_emb = model.encode(phrases)
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(phrase_emb)
    nmi_score = normalized_mutual_info_score(labels, kmeans.labels_)

    return nmi_score


def eval_retrieval(model: Embedder, kb_dataset: Dataset, test_dataset: Dataset):
    e_names = [x for x in kb_dataset["entity_name"] if x is not None]
    sen_embeddings = model.encode(e_names)

    emb_index = Reach(sen_embeddings, e_names)

    cnt, wrong_cnt = 0, 0
    mentions = test_dataset["query"]
    labels = test_dataset["label"]

    batch_emb = model.encode(mentions)

    I = emb_index.nearest_neighbor(batch_emb)
    predicted = [i[0][0] for i in I]
    for label, predict in zip(labels, predicted):
        cnt += 1
        if predict != label:
            wrong_cnt += 1
    acc = (cnt - wrong_cnt) * 1.0 / cnt

    return acc


def eval_single_autofj(dataset_name: str, model: Embedder):
    left_table, right_table, gt_table = load_data(dataset_name)
    left_table_list: list[str] = list(left_table.title)
    right_table_list: list[str] = list(right_table.title)
    left_label, right_label = list(gt_table.title_l), list(gt_table.title_r)
    gt_label = dict(zip(right_label, left_label))

    left_embs = normalize(model.encode(left_table_list))
    right_embs = normalize(model.encode(right_table_list))

    acc_cnt, total = 0, 0

    for index, r_t_emb in enumerate(right_embs):
        r_t = right_table_list[index]
        try:
            g_t = gt_label[r_t]
        except KeyError:
            continue

        score = r_t_emb @ left_embs.T
        pred_i = np.argmax(score)
        predicted = left_table_list[pred_i]

        if predicted == g_t:
            acc_cnt += 1
        total += 1
    return acc_cnt * 1.0 / total


def eval_autofj(model: Embedder, dataset: Dataset):
    table_names: list[str] = [row["Dataset"] for row in dataset]
    acc_list = []
    for table_name in table_names:
        acc_list.append(eval_single_autofj(dataset_name=table_name, model=model))

    return sum(acc_list) / len(acc_list)


def load_entity(entity_path: str) -> dict[str, list[str]]:
    e_names = []
    for line in open(entity_path, encoding="utf8"):
        e_names.append(line.strip())
    return {"mention": e_names, "entity": e_names}


def main() -> None:
    parser = get_default_argparser()
    args = parser.parse_args()

    embedder, name = load_embedder(args.input, args.word_level, args.model_path, args.device)

    ppdb_dataset = load_dataset("Lihuchen/pearl_benchmark", "ppdb", split="test")
    ppbd_score = eval_ppdb(embedder, ppdb_dataset)

    ppdb_filtered_dataset = load_dataset("Lihuchen/pearl_benchmark", "ppdb_filtered", split="test")
    ppbd_filtered_score = eval_ppdb(embedder, ppdb_filtered_dataset)

    turney_dataset = load_dataset("Lihuchen/pearl_benchmark", "turney", split="test")
    turney_score = eval_turney(embedder, turney_dataset)
    bird_dataset = load_dataset("Lihuchen/pearl_benchmark", "bird", split="test")
    bird_score = eval_bird(embedder, bird_dataset)

    yago_kb_dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split="yago")
    yago_test_dataset = load_dataset("Lihuchen/pearl_benchmark", "yago", split="test")
    yago_score = eval_retrieval(embedder, yago_kb_dataset, yago_test_dataset)

    umls_kb_dataset = load_dataset("Lihuchen/pearl_benchmark", "kb", split="umls")
    umls_test_dataset = load_dataset("Lihuchen/pearl_benchmark", "umls", split="umls")
    umls_score = eval_retrieval(embedder, umls_kb_dataset, umls_test_dataset)

    conll_dataset = load_dataset("Lihuchen/pearl_benchmark", "conll", split="test")
    conll_score = eval_clustering(embedder, conll_dataset, name="conll")

    bc5cdr_dataset = load_dataset("Lihuchen/pearl_benchmark", "bc5cdr", split="test")
    bc5cdr_score = eval_clustering(embedder, bc5cdr_dataset, name="bc5cdr")

    autofj_dataset = load_dataset("Lihuchen/pearl_benchmark", "autofj", split="test")
    autofj_score = eval_autofj(embedder, autofj_dataset)

    task_scores = {
        "ppdb": ppbd_score,
        "ppdb_filtered": ppbd_filtered_score,
        "turney": turney_score,
        "bird": bird_score,
        "yago": yago_score,
        "conll": conll_score,
        "bc5cdr": bc5cdr_score,
        "autofj": autofj_score,
        "umls": umls_score,
    }

    if args.suffix:
        name = f"{name}_{args.suffix}"

    # Create the results directory if it does not exist
    Path(f"results/{name}").mkdir(parents=True, exist_ok=True)

    # Save the scores json to a file
    with open(f"results/{name}/pearl_benchmark.json", "w") as file:
        json.dump(task_scores, file, indent=4)

    logger.info(task_scores)
    logger.info(f"Results saved to results/{name}/pearl_benchmark.json")


if __name__ == "__main__":
    setup_logging()
    main()
