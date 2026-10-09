# Training

Aside from [distillation](../../README.md#distillation), `model2vec` also supports fine-tuning static models with [pytorch](https://pytorch.org/). Every trainable model consists of the static embeddings of a `StaticModel`, optionally followed by a small MLP head. Both the embeddings and the head are trained.

There are four trainable models, one per task:

| Class | Task | Targets | Loss |
| --- | --- | --- | --- |
| `StaticModelForClassification` | Single- or multi-label classification | A label, or a list of labels, per text | (Binary) cross-entropy, or focal loss |
| `StaticModelForSimilarity` | Mapping texts to target vectors | A vector per text | Cosine distance |
| `StaticModelForRegression` | Regressing vectors of numbers | A vector per text | Mean squared error |
| `StaticModelForPairSimilarity` | Embedding related texts close together | A paired text per text | InfoNCE with in-batch negatives |

# Installation

To train, make sure you install the training extra:

```
pip install model2vec[train]
```

This installs `torch` and `datasets`.

# Quickstart

To train a model, initialize it from a pre-trained model, or from a `StaticModel`, for example one you distilled yourself:

```python
from model2vec.distill import distill
from model2vec.train import StaticModelForClassification

# From a pre-trained model: potion-base-32m is the default
classifier = StaticModelForClassification.from_pretrained(path="minishlab/potion-base-32M")

# From a distilled model
distilled_model = distill("baai/bge-base-en-v1.5")
classifier = StaticModelForClassification.from_static_model(model=distilled_model)
```

This creates a classifier with a single 512-unit hidden layer on top of the static embeddings. See [Model options](#model-options) for how to change this. The default for `from_pretrained` is [potion-base-32m](https://huggingface.co/minishlab/potion-base-32M), our best model to date. This is our recommended path if you're working with general English data.

Now let's train it. The example below uses the [`datasets`](https://github.com/huggingface/datasets) library, which is installed with the training extra.

```python
from datasets import load_dataset
from time import perf_counter

# Load the subj dataset
ds = load_dataset("setfit/subj")
train = ds["train"]
test = ds["test"]

s = perf_counter()
classifier = classifier.fit(train["text"], train["label"])

print(f"Training took {int(perf_counter() - s)} seconds.")
# Training took 31 seconds
classification_report = classifier.evaluate(test["text"], test["label"])
print(classification_report)
# Achieved 92.0 test accuracy
```

As you can see, we got a pretty nice 92% accuracy, with only 31 seconds of training.

The classes are taken from the labels you pass to `fit`, so labels can be strings or integers, and `out_dim` does not need to be set beforehand. After training, `classifier.classes` holds the classes in the order of the model's outputs.

Inference is as fast as you're used to from us:

```python
s = perf_counter()
classifier.predict(test["text"])
print(f"Took {int((perf_counter() - s) * 1000)} milliseconds for {len(test)} instances on CPU.")
# Took 66 milliseconds for 2000 instances on CPU.
```

`predict_proba` returns the probability of each class instead.

## Multi-label classification

Multi-label classification is supported out of the box. Just pass a list of lists to the `fit` function (e.g. `[[label1, label2], [label1, label3]]`), and a multi-label classifier will be trained. For example, the following code trains a multi-label classifier on the [go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset:

```python
from datasets import load_dataset
from model2vec.train import StaticModelForClassification

# Initialize a classifier from a pre-trained model
classifier = StaticModelForClassification.from_pretrained(path="minishlab/potion-base-32M")

# Load a multi-label dataset
ds = load_dataset("google-research-datasets/go_emotions")

# Inspect some of the labels
print(ds["train"]["labels"][40:50])
# [[0, 15], [15, 18], [16, 27], [27], [7, 13], [10], [20], [27], [27], [27]]

# Train the classifier on text (X) and labels (y)
classifier.fit(ds["train"]["text"], ds["train"]["labels"])
```

Then, we can evaluate the classifier:

```python
classification_report = classifier.evaluate(ds["test"]["text"], ds["test"]["labels"], threshold=0.3)
print(classification_report)
# {'accuracy': 0.41, 'macro avg': {'precision': 0.527, 'recall': 0.41, 'f1-score': 0.439, ...}, ...}
```

The `threshold` is the minimum probability for a label to be predicted, and can also be passed to `predict`. During training, the validation accuracy of a multi-label classifier is the Jaccard similarity between the predicted and true labels, at a threshold of 0.5.

The scores are competitive with the popular [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) model, while our model is orders of magnitude faster.

## Imbalanced data

If some classes are much rarer than others, there are two options, which can be combined.

`class_weight` weights the loss of each class. Pass `"balanced"` to weight each class by its inverse frequency, or a dict that maps each class to its weight:

```python
classifier.fit(X, y, class_weight="balanced")
classifier.fit(X, y, class_weight={"positive": 3.0, "negative": 1.0})
```

For multi-label classification, the weight of a class only applies to the texts that have that class.

`focal_gamma` switches the loss to a [focal loss](https://arxiv.org/abs/1708.02002), which down-weights the texts the model already classifies correctly with high confidence. The default, `0.0`, is equal to plain cross-entropy. A common value is `2.0`:

```python
classifier.fit(X, y, focal_gamma=2.0)
```

## Similarity and regression

`StaticModelForSimilarity` and `StaticModelForRegression` learn to map each text to a target vector. Pass the vectors as `y`, as a 2D array or tensor, a list of lists, or a dataset column that holds lists of numbers. The output dimension of the model is set to the dimension of the vectors.

`StaticModelForSimilarity` is trained with a cosine distance loss, so only the direction of the output matters. A typical use is to train a static model to mimic the embeddings of a larger model:

```python
from sentence_transformers import SentenceTransformer
from model2vec.train import StaticModelForSimilarity

teacher = SentenceTransformer("baai/bge-base-en-v1.5")
targets = teacher.encode(texts)

model = StaticModelForSimilarity.from_pretrained(path="minishlab/potion-base-32M")
model.fit(texts, targets)
```

`StaticModelForRegression` is trained with a mean squared error loss, so the output has to match the targets exactly. To regress a single number, pass a vector of length one for each text, e.g. `[[0.3], [1.2], ...]`.

Both models stop early on the validation loss.

## Pair similarity

`StaticModelForPairSimilarity` trains a model to embed pairs of related texts (e.g. queries and their matching documents) close together, by encoding both sides with the same model. It is trained with an InfoNCE loss with in-batch negatives: each `text_a` is pulled towards its paired `text_b` and pushed away from every other `text_b` in the batch:

```python
from model2vec.train import StaticModelForPairSimilarity

model = StaticModelForPairSimilarity.from_pretrained(path="minishlab/potion-base-32M", n_layers=0)
model.fit(text_a=queries, text_b=documents)

static_model = model.to_static_model()
```

With `n_layers=0`, the model has no head, so the trained embeddings can be turned back into a regular `StaticModel`. With the default `n_layers=1`, the model has an MLP head, and you need [`to_pipeline`](#using-a-trained-model) to keep it.

Because the other pairs in a batch serve as negatives, the training and validation sets each need at least two pairs, and the batch size must be at least two. Larger batches give more negatives per pair. Pairs with the same `text_a`, or the same `text_b`, are not used as negatives for each other.

The InfoNCE temperature can be set with `temperature` (default `0.05`). It must be positive.

The validation set is passed as `text_a_val` and `text_b_val`, instead of `X_val` and `y_val`.

# Model options

All four models are created with `from_pretrained` or `from_static_model`, which take the same keyword arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `n_layers` | `1` | The number of hidden layers in the head. |
| `hidden_dim` | `512` | The size of the hidden layers. |
| `out_dim` | | The output dimension. For classification, similarity and regression, this is set by `fit`. For pair similarity, it defaults to the embedding dimension. |
| `max_length` | The static model's | The maximum number of tokens per text, for both training and inference. Pass `None` to disable truncation. |
| `freeze` | `False` | Freeze the token embeddings, so that only the head and token weights are trained. |
| `freeze_weights` | `None` | Controls the token weights. `None` trains the model's own token weights, and leaves a model without token weights without them. `False` also gives a model without token weights learnable weights, starting at 1. `True` freezes them. |
| `normalize` | `True` | Normalize the embeddings before the head. |

With `n_layers=0`, a classifier gets a single linear layer. A similarity or pair model gets no head at all if `out_dim` equals the embedding dimension, and a single linear layer otherwise.

# Training options

`fit` takes the following options, which are shared by all models, unless noted otherwise:

| Argument | Default | Description |
| --- | --- | --- |
| `learning_rate` | `1e-3` | The learning rate of the Adam optimizer. |
| `batch_size` | `None` | The batch size. If `None`, a multiple of 32 between 32 and 512 is chosen based on the size of the training set. |
| `min_epochs` | `None` | The minimum number of epochs before early stopping can stop training. |
| `max_epochs` | `-1` | The maximum number of epochs. If `-1`, training continues until early stopping triggers. |
| `early_stopping_patience` | `5` | The number of validation checks without improvement before training stops. If `None`, early stopping is disabled. |
| `test_size` | `0.1` | The size of the validation split, if no validation set is passed. See [Validation](#validation). |
| `X_val`, `y_val` | `None` | An explicit validation set. `text_a_val` and `text_b_val` for pair similarity. |
| `validation_steps` | `None` | Validate every this many training steps. See [Validation](#validation). |
| `device` | `"auto"` | The device to train on. `"auto"` picks CUDA, then MPS, then CPU. |
| `random_seed` | `42` | The seed for initialization and shuffling. The validation split always uses a seed of 42. |
| `token_dropout` | `0.0` | The fraction of tokens to randomly drop from each training text. This is a form of data augmentation, and has no effect during validation. Must be in `[0, 1)`. |
| `class_weight` | `None` | Classification only. See [Imbalanced data](#imbalanced-data). |
| `focal_gamma` | `0.0` | Classification only. See [Imbalanced data](#imbalanced-data). |
| `temperature` | `0.05` | Pair similarity only. The temperature of the InfoNCE loss. |

Calling `fit` re-initializes the head, the embeddings and the token weights from the static model, so calling it twice does not continue training.

## Validation

Without an explicit validation set, `fit` holds out `test_size` of the data for validation, capped at 10,000 rows; pass an int to hold out an exact number of rows. For single-label classification, the split is stratified by class.

By default, the model is validated once per epoch. If an epoch has more than 250 batches, it is instead validated about four times per epoch, but at most once every 250 steps. `validation_steps` sets a fixed number of steps between validations instead.

Classifiers stop early on the validation accuracy, the other models on the validation loss. Whenever the validation loss stops improving for a few epochs, the learning rate is halved. After training, the weights from the best validation check are loaded back into the model.

The training loop shows the loss, and for classifiers the accuracy, of the latest training batches and of the latest validation check in the progress bar.

## Large datasets

Training data is read and tokenized per batch, so `fit` also accepts the columns of a Hugging Face dataset. These are not loaded into memory:

```python
from datasets import load_dataset

dataset = load_dataset("sentence-transformers/gooaq", split="train")
model.fit(text_a=dataset["question"], text_b=dataset["answer"])
```

Because batches are shuffled, training reads the rows of a dataset in random order. If the dataset is stored on disk and does not fit in memory, this can be slow, especially on a network file system.

Columns of a dataset with a transform, set with `with_transform`, are not accepted. Apply the transform first with `dataset.map(transform, batched=True)`. For a dataset loaded from disk or the Hub, this writes the result to the cache on disk, so it is still not loaded into memory.

Columns of an iterable dataset, such as one loaded with `streaming=True`, are not accepted. Neither is a whole `Dataset`: pass its columns, such as `dataset["text"]`.

# Using a trained model

You can turn any trained model into a lightweight inference pipeline:

```python
pipeline = classifier.to_pipeline()
```

This strips away `torch`: the head's weights are plain `numpy` arrays, so the resulting `StaticModelPipeline` can be used for inference without installing `torch`. For a classifier, `pipeline.predict` returns labels, and `pipeline.predict_proba` and `pipeline.evaluate` work as on the classifier. For the other models, `pipeline.predict` returns the output vectors.

If you want to persist your pipeline locally, or to the Hugging Face Hub, you can use our built-in functions:

```python
pipeline.save_pretrained(path)
pipeline.push_to_hub("my_cool/project")
```

Later, you can load these as follows:

```python
from model2vec.inference import StaticModelPipeline

pipeline = StaticModelPipeline.from_pretrained("my_cool/project")
```

Loading pipelines in this way is _extremely_ fast. It takes only 30ms to load a pipeline from disk.

A pipeline can also be exported to ONNX, including its head:

```python
from model2vec.onnx import export_model_to_onnx

export_model_to_onnx(pipeline, "my_onnx_model")
```

`to_static_model` turns a trained model into a regular `StaticModel`, without its head, but with the trained embeddings and token weights. This is mainly useful for models without a head, such as a pair similarity model with `n_layers=0`.

# Bring your own architecture

Our training architecture is set up to be extensible, with each task having a specific class. All of them subclass `BaseFinetuneable` (in [`base.py`](base.py)), which contains the shared functionality:

* `construct_head`: constructs the head on top of the static model. For example, if you want to create a model that has LayerNorm, just subclass, and replace this function. This should be the main function to update if you want to change model behavior.
* `construct_embeddings` and `construct_weights`: construct the token embeddings and the token weights.
* `_encode`: the encoding function used in the model: a weighted mean over the token embeddings, followed by normalization.
* `_create_datasets`: splits off the validation data, and creates the datasets that are tokenized per batch during training.
* `_to_targets`: turns a batch of labels into the targets of the loss.
* `fit`: contains all the fitting logic, and is implemented by each task.

The training loop itself is defined in `model2vec.train.trainer.run_training_loop`, a plain torch loop that is fairly basic and easy to modify. Each task passes in its own loss function, and, for classification, a small function that computes extra metrics like accuracy. `StaticModelForRegression`, for example, only replaces the loss function of `StaticModelForSimilarity`.

# Results

We ran extensive benchmarks where we compared our model to several well known architectures. The results can be found in the [training results](https://github.com/MinishLab/model2vec/tree/main/results#training-results) documentation.
