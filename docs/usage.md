
# Usage

This document provides an overview of how to use Model2Vec for inference, distillation, training, and evaluation.

## Table of Contents
- [Inference](#inference)
  - [Inference with a pretrained model](#inference-with-a-pretrained-model)
  - [Inference with the Sentence Transformers library](#inference-with-the-sentence-transformers-library)
- [Distillation](#distillation)
    - [Distilling from a Sentence Transformer](#distilling-from-a-sentence-transformer)
    - [Distilling from a loaded model](#distilling-from-a-loaded-model)
    - [Distilling with the Sentence Transformers library](#distilling-with-the-sentence-transformers-library)
    - [Distilling with a custom vocabulary](#distilling-with-a-custom-vocabulary)
- [Training](#training)
    - [Training a classifier](#training-a-classifier)
- [Evaluation](#evaluation)
    - [Installation](#installation)
    - [Evaluation Code](#evaluation-code)

## Inference

### Inference with a pretrained model

Inference works as follows. The example shows one of our own models, but you can also just load a local one, or another one from the hub.
```python
from model2vec import StaticModel

# Load a model from the Hub. You can optionally pass a token when loading a private model
model = StaticModel.from_pretrained(model_name="minishlab/potion-base-8M", token=None)

# Make embeddings
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])

# Make sequences of token embeddings
token_embeddings = model.encode_as_sequence(["It's dangerous to go alone!", "It's a secret to everybody."])
```

### Inference with the Sentence Transformers library

The following code snippet shows how to use a Model2Vec model in the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) library. This is useful if you want to use the model in a Sentence Transformers pipeline.

```python
from sentence_transformers import SentenceTransformer

# Load a Model2Vec model from the Hub
model = SentenceTransformer("minishlab/potion-base-8M")

# Make embeddings
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
```

## Distillation

### Distilling from a Sentence Transformer

The following code can be used to distill a model from a Sentence Transformer. As mentioned above, this leads to really small model that might be less performant.
```python
from model2vec.distill import distill

# Distill a Sentence Transformer model
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5", pca_dims=256)

# Save the model
m2v_model.save_pretrained("m2v_model")

```

### Distilling from a loaded model

If you already have a model loaded, or need to load a model in some special way, we also offer an interface to distill models in memory.

```python
from transformers import AutoModel, AutoTokenizer

from model2vec.distill import distill_from_model

# Assuming a loaded model and tokenizer
model_name = "baai/bge-base-en-v1.5"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

m2v_model = distill_from_model(model=model, tokenizer=tokenizer, pca_dims=256)

m2v_model.save_pretrained("m2v_model")

```

### Distilling with the Sentence Transformers library

The following code snippet shows how to distill a model using the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) library. This is useful if you want to use the model in a Sentence Transformers pipeline.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

static_embedding = StaticEmbedding.from_distillation("BAAI/bge-base-en-v1.5", device="cpu", pca_dims=256)
model = SentenceTransformer(modules=[static_embedding])
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
```

### Distilling with a custom vocabulary

If you pass a vocabulary, you get a set of static word embeddings, together with a custom tokenizer for exactly that vocabulary. This is comparable to how you would use GLoVe or traditional word2vec, but doesn't actually require a corpus or data.
```python
from model2vec.distill import distill

# Load a vocabulary as a list of strings
vocabulary = ["word1", "word2", "word3"]

# Distill a Sentence Transformer model with the custom vocabulary
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5", vocabulary=vocabulary)

# Save the model
m2v_model.save_pretrained("m2v_model")

# Or push it to the hub
m2v_model.push_to_hub("my_organization/my_model", token="<it's a secret to everybody>")
```

By default, this will distill a model with a subword tokenizer, combining the models (subword) vocab with the new vocabulary. If you want to get a word-level tokenizer instead (with only the passed vocabulary), the `use_subword` parameter can be set to `False`, e.g.:

```python
m2v_model = distill(model_name=model_name, vocabulary=vocabulary, use_subword=False)
```

**Important note:** we assume the passed vocabulary is sorted in rank frequency. i.e., we don't care about the actual word frequencies, but do assume that the most frequent word is first, and the least frequent word is last. If you're not sure whether this is case, set `apply_zipf` to `False`. This disables the weighting, but will also make performance a little bit worse.

### Quantization

Models can be quantized to `float16` (default) or `int8` during distillation, or when loading from disk.

```python
from model2vec.distill import distill

# Distill a Sentence Transformer model and quantize is to int8
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5", quantize_to="int8")

# Save the model. This model is now 25% of the size of a normal model.
m2v_model.save_pretrained("m2v_model")
```

You can also quantize during loading.

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8m", quantize_to="int8")
```

### Dimensionality reduction

Because almost all Model2Vec models have been distilled using PCA, and because PCA explicitly orders dimensions from most informative to least informative, we can perform dimensionality reduction during loading. This is very similar to how matryoshka embeddings work.

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8m", dimensionality=32)

print(model.embedding.shape)
# (29528, 32)
```

### Combining quantization and dimensionality reduction

Combining these tricks can lead to extremely small models. For example, using this, we can reduce the size of `potion-base-8m`, which is now 30MB, to only 1MB:

```python
model = StaticModel.from_pretrained("minishlab/potion-base-8m",
                                    dimensionality=32,
                                    quantize_to="int8")
print(model.embedding.nbytes)
# 944896 bytes = 944kb
```

This should be enough to satisfy even the strongest hardware constraints.

## Training

### Training a classifier

Model2Vec can be used to train a classifier on top of a distilled model. The following code snippet shows how to train a classifier on top of a distilled model. For more advanced usage, as well as results, please refer to the [training documentation](https://github.com/MinishLab/model2vec/blob/main/model2vec/train/README.md).

```python
import numpy as np
from datasets import load_dataset
from model2vec.train import StaticModelForClassification

# Initialize a classifier from a pre-trained model
classifer = StaticModelForClassification.from_pretrained("minishlab/potion-base-8M")

# Load a dataset
ds = load_dataset("setfit/subj")
train = ds["train"]
test = ds["test"]

X_train, y_train = train["text"], train["label"]
X_test, y_test = test["text"], test["label"]

# Train the classifier
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_hat = classifier.predict(X_test)
accuracy = np.mean(np.array(y_hat) == np.array(y_test)) * 100
```

## Evaluation

### Installation

Our models can be evaluated using our [evaluation package](https://github.com/MinishLab/evaluation). Install the evaluation package with:

```bash
pip install git+https://github.com/MinishLab/evaluation.git@main
```

### Evaluation Code

The following code snippet shows how to evaluate a Model2Vec model:
```python
from model2vec import StaticModel

from evaluation import CustomMTEB, get_tasks, parse_mteb_results, make_leaderboard, summarize_results
from mteb import ModelMeta

# Get all available tasks
tasks = get_tasks()
# Define the CustomMTEB object with the specified tasks
evaluation = CustomMTEB(tasks=tasks)

# Load the model
model_name = "m2v_model"
model = StaticModel.from_pretrained(model_name)

# Optionally, add model metadata in MTEB format
model.mteb_model_meta = ModelMeta(
            name=model_name, revision="no_revision_available", release_date=None, languages=None
        )

# Run the evaluation
results = evaluation.run(model, eval_splits=["test"], output_folder=f"results")

# Parse the results and summarize them
parsed_results = parse_mteb_results(mteb_results=results, model_name=model_name)
task_scores = summarize_results(parsed_results)

# Print the results in a leaderboard format
print(make_leaderboard(task_scores))
```
