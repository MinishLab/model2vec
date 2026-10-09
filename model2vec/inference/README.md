# Inference

This subpackage contains `StaticModelPipeline`, which runs a trained model: a `StaticModel` followed by a small MLP head. The head is implemented in `numpy`, so a pipeline only needs the base `model2vec` package: no `torch` or `scikit-learn`.

A pipeline is created from any of the trainable models in [`model2vec.train`](../train/README.md). For a classifier, the pipeline predicts labels. For the other models, it predicts vectors.

# Creating a pipeline

To create a pipeline, train a model and call `to_pipeline`:

```python
from datasets import load_dataset
from model2vec.train import StaticModelForClassification

ds = load_dataset("setfit/subj")
classifier = StaticModelForClassification.from_pretrained(path="minishlab/potion-base-32M")
classifier.fit(ds["train"]["text"], ds["train"]["label"])

pipeline = classifier.to_pipeline()
```

# Saving and loading

Pipelines can be saved to a folder, or pushed to the Hugging Face Hub:

```python
pipeline.save_pretrained("my_pipeline")
pipeline.push_to_hub("my_cool/project")
```

`push_to_hub` also takes a `token`, a `subfolder` to push to, and `private=True` to create a private repository.

Later, you can load these as follows:

```python
from model2vec.inference import StaticModelPipeline

pipeline = StaticModelPipeline.from_pretrained("my_cool/project")
```

To load a private pipeline, pass a `token`. Loading a pipeline is _extremely_ fast: it takes only 30ms to load one from disk.

A saved pipeline is a regular model2vec model, with two additions: the weights of the head are stored in `head.safetensors`, and the head's activation and classes are stored under `head_config` in `config.json`. This means the folder can also be loaded as a plain `StaticModel`, which gives you the embeddings without the head.

# Usage

`predict` takes a list of texts, or a single text:

```python
pipeline.predict(["This movie was great.", "The plot follows a detective in Paris."])
# array([0, 1])
```

What it returns depends on the head:

* **Single-label classification**: an array with one label per text. The labels have the same type as the labels the model was trained on.
* **Multi-label classification**: an object array with an array of labels per text. A label is predicted if its probability is higher than `threshold`, which defaults to `0.5`.
* **Similarity, regression and pair similarity**: an array with one output vector per text.

For classifiers, `predict_proba` returns the probability of each class, in the order of `pipeline.classes_`. These are softmax probabilities for single-label classification, and sigmoid probabilities for multi-label classification.

`evaluate` predicts labels for a set of texts, and compares them with the true labels:

```python
report = pipeline.evaluate(ds["test"]["text"], ds["test"]["label"])
# {'0': {'precision': ..., 'recall': ..., 'f1-score': ..., 'support': ...}, ..., 'accuracy': ..., 'macro avg': {...}, 'weighted avg': {...}}
```

The report has the same format as the dictionary returned by scikit-learn's `classification_report`, but does not need scikit-learn. For multi-label classification, accuracy is the fraction of texts for which all labels are predicted correctly, and `threshold` can be passed as with `predict`. The same function is available as `evaluate_single_or_multi_label`, which takes predictions and true labels.

`predict_proba` and `evaluate` raise an error for pipelines that predict vectors.

`predict` and `predict_proba` take the same encoding options as `StaticModel.encode`: `batch_size`, `max_length`, `show_progress_bar`, `use_multiprocessing` and `multiprocessing_threshold`. If `max_length` is not passed, the model's own `max_length` is used. Pass `max_length=None` to disable truncation.

The underlying `StaticModel` is available as `pipeline.model`, and the head as `pipeline.head`.

# ONNX export

A pipeline can be exported to ONNX, including its head:

```python
from model2vec.onnx import export_model_to_onnx

export_model_to_onnx(pipeline, "my_onnx_pipeline")
```

# Migrating a legacy pipeline

Pipelines saved by older versions of model2vec store the head as a `scikit-learn`/`skops` `pipeline.skops` file instead of `head.safetensors`. An example is our [potion-edu classifier](https://huggingface.co/minishlab/potion-8m-edu-classifier). `from_pretrained` still loads these automatically, falling back to the legacy format and emitting a warning. This requires `scikit-learn` and `skops` to be installed.

To upgrade a pipeline to the current format (and silence the warning), convert it with `convert_legacy_pipeline` and save the result:

```python
from model2vec.inference import convert_legacy_pipeline

pipeline = convert_legacy_pipeline("minishlab/potion-8m-edu-classifier")
pipeline.save_pretrained("potion-8m-edu-classifier")
```

Only heads that are a scikit-learn `MLPClassifier` or `MLPRegressor` with ReLU activations can be converted. By default, the `skops` file may only contain `scikit-learn` types. Pass `trust_remote_code=True` to load other types, but only for files you trust.
