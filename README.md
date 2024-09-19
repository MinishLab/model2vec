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
- **Small**: Model2Vec can reduce the size of a Sentence Transformer model by a factor of .
- **Fast distillation**: Model2Vec can distill a Sentence Transformer model in X minutes.
- **Fast inference**: Model2Vec creates static embeddings that are up to X times faster than the original model.
- **No data needed**: Distillation happens directly on a token leven, so no dataset is needed.
- **Simple to use**: Model2Vec provides an easy to use interface for distilling and inferencing Model2Vec models.
- **Bring your own model**: Model2Vec can be applied to any Sentence Transformer model.
- **Bring your own vocabulary**: Model2Vec can be applied to any vocabulary, allowing you to use your own domain-specific vocabulary.
- **Multi-lingual**: Model2Vec can easily be applied to any language.
- **Easy Evaluation**: Model2Vec comes with a set of evaluation tasks to measure the performance of the distilled model.

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
Model2Vec allows anyone to create their own static embeddings from any Sentence Transformer model in minutes. It can easily be applied to other languages by using a language-specific Sentence Transformer model and vocab. Similarly, it can be applied to specific domains by using a domain specific model, vocab, or both. This makes it an ideal tool for fast prototyping, research, and production use cases where speed and size are more important than performance.

## Results

### Main Results

Model2Vec is evaluated on MTEB, as well as two additional tasks: PEARL (a phrase representation task) and WordSim (a word similarity task). The results are shown in the table below.




| Model            | Avg (All)   | Avg (MTEB) | Class | Clust | PairClass | Rank  | Ret   | STS   | Sum   | PEARL | WordSim |
|------------------|-------------|------------|-------|-------|-----------|-------|-------|-------|-------|-------|---------|
| all-MiniLM-L6-v2 | 56.08       | 56.09      | 62.62 | 41.94 | 82.37     | 58.04 | 41.95 | 78.90 | 30.81 | 60.83 | 49.91   |
| M2V_base_glove   | 48.58       | 47.60      | 61.35 | 30.52 | 75.34     | 48.50 | 29.26 | 70.31 | 31.50 | 50.28 | 54.29   |
| M2V_base_output  | 46.79       | 45.34      | 61.25 | 25.58 | 74.90     | 47.63 | 26.14 | 68.58 | 29.20 | 54.02 | 49.18   |
| GloVe_300d       | 42.84       | 42.36      | 57.31 | 27.66 | 72.48     | 43.30 | 22.78 | 61.90 | 28.81 | 45.65 | 43.05   |
| WL256*           | 48.88       | 49.36      | 58.98 | 33.34 | 74.00     | 52.03 | 33.12 | 73.34 | 29.05 | 48.81 | 45.16   |

For readability, the mteb task names are abbreviated as follows:
- Class: Classification
- Clust: Clustering
- PairClass: PairClassification
- Rank: Reranking
- Ret: Retrieval
- STS: Semantic Textual Similarity
- Sum: Summarization

* WL256, introduced in the [WordLlama](https://github.com/dleemiller/WordLlama/tree/main) package is included for comparison. However, we believe it is heavily overfit to the MTEB dataset due to the training data used for the model. This can be seen by the fact that the WL256 model performs much worse on the non MTEB tasks (PEARL and WordSim) than our models. The results shown in the [Classification and Speed Benchmarks](#classification-and-speed-benchmarks) further support this.

### Classification and Speed Benchmarks

In addition to the MTEB evaluation, Model2Vec is evaluated on a number of classification datasets. These are used as additional analysis to avoid overfitting to the MTEB dataset and to benchmark the speed of the model. The results are shown in the table below.

| model            |   Average |     sst2 |   imdb |     trec |   ag_news |
|:-----------------|----------:|---------:|-------:|---------:|----------:|
| bge-base-en-v1.5 |  0.900079 | 0.915367 | 0.9188 | 0.851648 |  0.9145   |
| all-MiniLM-L6-v2 |  0.840987 | 0.839495 | 0.8136 | 0.813187 |  0.897667 |
| M2V_base_output  |  0.822326 | 0.809206 | 0.8456 | 0.752747 |  0.88175  |
| M2V_base_glove   |  0.807597 | 0.830735 | 0.8524 | 0.661172 |  0.886083 |
| WL256            |  0.78479  | 0.76882  | 0.8012 | 0.692308 |  0.876833 |
| GloVe_300d       |  0.77768  | 0.816778 | 0.84   | 0.556777 |  0.897167 |

As can be seen, the Model2Vec models outperforms the GloVe and WL256 models on all classification tasks, and is competitive with the all-MiniLM-L6-v2 model while being much faster.

The scatterplot below shows the relationship between the number of sentences per second and the average classification score. The bubble sizes correspond to the number of parameters in the models (larger = more parameters), and the colors correspond to the sentences per second (greener = more sentences per second). This plot shows that the Model2Vec models are much faster than the other models, while still being competitive in terms of classification performance.

![Description](assets/images/sentences_per_second_vs_average_score.png)

Picture
Table
## Roadmap

## Citing
