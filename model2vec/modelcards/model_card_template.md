---
{{ card_data }}
---

# {{ model_name }} Model Card

This [Model2Vec](https://github.com/MinishLab/model2vec) model is a distilled version of {% if base_model %}the {{ base_model }}(https://huggingface.co/{{ base_model }}){% else %}a{% endif %} Sentence Transformer. It uses static embeddings, allowing text embeddings to be computed orders of magnitude faster on both GPU and CPU. It is designed for applications where computational resources are limited or where real-time performance is critical. Model2Vec models are the smallest, fastest, and most performant static embedders available. The distilled models are up to 50 times smaller and 500 times faster than traditional Sentence Transformers.


## Installation

Install model2vec using pip:
```
pip install model2vec
```

## Usage

### Using Model2Vec

The [Model2Vec library](https://github.com/MinishLab/model2vec) is the fastest and most lightweight way to run Model2Vec models.

Load this model using the `from_pretrained` method:
```python
from model2vec import StaticModel

# Load a pretrained Model2Vec model
model = StaticModel.from_pretrained("{{ model_name }}")

# Compute text embeddings
embeddings = model.encode(["Example sentence"])
```

### Using Sentence Transformers

You can also use the [Sentence Transformers library](https://github.com/UKPLab/sentence-transformers) to load and use the model:

```python
from sentence_transformers import SentenceTransformer

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("{{ model_name }}")

# Compute text embeddings
embeddings = model.encode(["Example sentence"])
```

### Distilling a Model2Vec model

You can distill a Model2Vec model from a Sentence Transformer model using the `distill` method. First, install the `distill` extra with `pip install model2vec[distill]`. Then, run the following code:

```python
from model2vec.distill import distill

# Distill a Sentence Transformer model, in this case the BAAI/bge-base-en-v1.5 model
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5", pca_dims=256)

# Save the model
m2v_model.save_pretrained("m2v_model")
```

## How it works

Model2vec creates a small, fast, and powerful model that outperforms other static embedding models by a large margin on all tasks we could find, while being much faster to create than traditional static embedding models such as GloVe. Best of all, you don't need any data to distill a model using Model2Vec.

It works by passing a vocabulary through a sentence transformer model, then reducing the dimensionality of the resulting embeddings using PCA, and finally weighting the embeddings using [SIF weighting](https://openreview.net/pdf?id=SyK00v5xx). During inference, we simply take the mean of all token embeddings occurring in a sentence.

## Additional Resources

- [Model2Vec Repo](https://github.com/MinishLab/model2vec)
- [Model2Vec Base Models](https://huggingface.co/collections/minishlab/model2vec-base-models-66fd9dd9b7c3b3c0f25ca90e)
- [Model2Vec Results](https://github.com/MinishLab/model2vec/tree/main/results)
- [Model2Vec Tutorials](https://github.com/MinishLab/model2vec/tree/main/tutorials)
- [Website](https://minishlab.github.io/)


## Library Authors

Model2Vec was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of [Stephan Tulkens](https://github.com/stephantul) and [Thomas van Dongen](https://github.com/Pringled).

## Citation

Please cite the [Model2Vec repository](https://github.com/MinishLab/model2vec) if you use this model in your work.
```
@article{minishlab2024model2vec,
  author = {Tulkens, Stephan and {van Dongen}, Thomas},
  title = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}
```
