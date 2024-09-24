---
model_name: {model_name}
base_model: {base_model}
language: {language}
library_name: 'model2vec'
license: {license}
tags: [embeddings, sentence-transformers, static-embeddings]
---

# {model_name} Model Card

Model2Vec distills a Sentence Transformer into a small, static model.
This model is ideal for applications requiring fast, lightweight embeddings.



## Installation

Install model2vec using pip:
```
pip install model2vec
```

## Usage
A StaticModel can be loaded using the `from_pretrained` method:
```python
from model2vec import StaticModel
model = StaticModel.from_pretrained("minishlab/M2V_base_output")
embeddings = model.encode(["Example sentence"])
```

Alternatively, you can distill your own model using the `distill` method:
```python
from model2vec.distill import distill

# Choose a Sentence Transformer model
model_name = "BAAI/bge-base-en-v1.5"

# Distill the model
m2v_model = distill(model_name=model_name, pca_dims=256)

# Save the model
m2v_model.save_pretrained("m2v_model")
```

## How it works

Model2vec creates a small, fast, and powerful model that outperforms other static embedding models by a large margin on all tasks we could find, while being much faster to create than traditional static embedding models such as GloVe. Best of all, you don't need any data to distill a model using Model2Vec.

It works by passing a vocabulary through a sentence transformer model, then reducing the dimensionality of the resulting embeddings using PCA, and finally weighting the embeddings using zipf weighting. During inference, we simply take the mean of all token embeddings occurring in a sentence.

## Citation

Please cite the [Model2Vec repository](https://github.com/MinishLab/model2vec) if you use this model in your work.

## Additional Resources

- [Model2Vec Repo](https://github.com/MinishLab/model2vec)
- [Model2Vec Results](https://github.com/MinishLab/model2vec?tab=readme-ov-file#results)
- [Model2Vec Tutorials](https://github.com/MinishLab/model2vec/tree/main/tutorials)

## Model Authors

Model2Vec was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of Stephan Tulkens and Thomas van Dongen.
