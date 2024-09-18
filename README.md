# Model2Vec: Turn any Sentence Transformer into a Small Fast Model


**Model2Vec** is a method to turn any Sentence Transformer model into a small, fast model.

## Table of Contents
- [Main Features](#main-features)
- [Installation](#installation)
- [Useage](#useage)
- [What is Model2Vec?](#what-is-model2vec)
- [Who is this for?](#who-is-this-for)
- [Results](#results)
- [Roadmap](#roadmap)
- [Citing](#citing)

## Main Features
- **Small**: Model2Vec can reduce the size of a Sentence Transformer model by a factor of X.
- **Fast distillation**: Model2Vec can distill a Sentence Transformer model in X.
- **Fast inference**: Model2Vec creates static embeddings that are up to X times faster than the original model.
- **Simple**: Model2Vec is easy to use and can be applied to any Sentence Transformer model, requiring only a few lines of code.
- **Bring your own model**: Model2Vec can be applied to any Sentence Transformer model.
- **Bring your own vocabulary**: Model2Vec can be applied to any vocabulary, allowing you to use your own domain-specific vocabulary.
- **Multi-lingual**: Model2Vec can easily be applied to any language.
- **Evaluation**: Model2Vec comes with a set of evaluation tasks to measure the performance of the distilled model.

## Installation
```bash
pip install model2vec
```
TODO: add optional installation for evaluation/reproduction of results

## Useage

### Distilling a Model2Vec model
```python
from model2vec import ...
```

Alternatively, you can use the command line interface:
```bash
python3 -m model2vec.distill --model-name BAAI/bge-base-en-v1.5 --vocabulary-path vocab.txt --device mps --save-path model2vec_model
```

### Inferencing a Model2Vec model
```python
from model2vec import StaticEmbedder

model_name = "model2vec_model"
model = StaticEmbedder.from_pretrained(model_name)

# Get embeddings
embeddings = model.encode["It's dangerous to go alone!", "It's a secret to everyone."]
```

### Evaluating a Model2Vec model
```python
from evaluation import CustomMTEB, get_tasks, parse_mteb_results, make_leaderboard, summarize_results
from mteb import ModelMeta

# Get all available tasks
tasks = get_tasks()
# Define the CustomMTEB object with the specified tasks
evaluation = CustomMTEB(tasks=tasks)

# Load the model
model_name = "model2vec_model"
model = StaticEmbedder.from_pretrained(model_name)

# Optionally, add model metadata in MTEB format
model.mteb_model_meta = ModelMeta(
            name=model_name, revision="no_revision_available", release_date=None, languages=None
        )

# Run the evaluation
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results/{model_name}")

# Parse the results and summarize them
parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
task_scores = summarize_results(parsed_results)
# Print the results in a leaderboard format
print(make_leaderboard(task_scores))
```

## What is Model2Vec?
Model2Vec is a simple and effective method to turn any sentence transformer into static embeddings. It works by inferencing a vocabulary with the specified Sentence Transformer model, reducing the dimensionality of the embeddings using PCA, weighting the embeddings using zipf weighting, and storing the embeddings in a static format.

This technique creates a small, fast, and powerful model that outperforms other static embedding models by a large margin on a a number of relevent tasks, while being much faster to create than traditional static embedding models such as Glove.


## Who is this for?

## Results

```markdown
| Model                                         |   Average (All) |   Average (MTEB) |   Classification |   Clustering |   PairClassification |   Reranking |   Retrieval |   STS |   Summarization |   PEARL |   WordSim |
|:----------------------------------------------|----------------:|-----------------:|-----------------:|-------------:|---------------------:|------------:|------------:|------:|----------------:|--------:|----------:|
| WL256                                         |          nan    |           nan    |            58.98 |        33.34 |                74    |       52.03 |      nan    | 73.34 |           29.05 |   48.81 |     45.16 |
| model2vec-bge_base-glove-zipf-pca             |           49.08 |            47.86 |            61.89 |        30.36 |                75.64 |       48.6  |       29.71 | 70.59 |           30.78 |   52.66 |     54.2  |
| model2vec-bge_large-glove-zipf-pca            |           48.94 |            47.56 |            62.48 |        30.02 |                75.41 |       47.9  |       28.72 | 70.31 |           31.61 |   51.51 |     56.65 |
| model2vec-bge_base-glove-zipf                 |           46.69 |            45.5  |            62.81 |        23.46 |                72.11 |       45.72 |       29.56 | 66.29 |           30.97 |   50.23 |     51.67 |
| model2vec-bge_base-output_embeddings-zipf-pca |           45.27 |            43.34 |            60.64 |        23.14 |                74.62 |       46.61 |       23.22 | 65.68 |           29.35 |   54.16 |     49.25 |
| model2vec-bge_base-glove-pca                  |           43.32 |            40.91 |            60.68 |        23.96 |                66.23 |       45.11 |       19.43 | 59.83 |           30.48 |   49.84 |     54.22 |
| komninos                                      |           43.11 |            42.86 |            57.7  |        28.86 |                73    |       44.75 |       22.45 | 62.52 |           30.5  |   46.63 |     40.54 |
| glove                                         |           42.84 |            42.36 |            57.31 |        27.66 |                72.48 |       43.3  |       22.78 | 61.9  |           28.81 |   45.65 |     43.05 |
| model2vec-bge_base-glove                      |           40.54 |            38.2  |            60.99 |        21.4  |                57.54 |       41.9  |       17.98 | 53.14 |           30.87 |   46.32 |     51.85 |
```

## Roadmap

## Citing
