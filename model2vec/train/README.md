# Training

Aside from [distillation](../../README.md#distillation), `model2vec` also supports training simple classifiers on top of static models, using [pytorch](https://pytorch.org/), [lightning](https://lightning.ai/) and [scikit-learn](https://scikit-learn.org/stable/index.html).

We support both single and multi-label classification, which work seamlessly based on the labels you provide.

# Installation

To train, make sure you install the training extra:

```
pip install model2vec[training]
```

# Quickstart

To train a model, simply initialize it using a `StaticModel`, or from a pre-trained model, as follows:

```python
from model2vec.distill import distill
from model2vec.train import StaticModelForClassification

# From a distilled model
distilled_model = distill("baai/bge-base-en-v1.5")
classifier = StaticModelForClassification.from_static_model(model=distilled_model)

# From a pre-trained model: potion is the default
classifier = StaticModelForClassification.from_pretrained(model_name="minishlab/potion-base-32m")
```

This creates a very simple classifier: a StaticModel with a single 512-unit hidden layer on top. You can adjust the number of hidden layers and the number units through some parameters on both functions. Note that the default for `from_pretrained` is [potion-base-32m](https://huggingface.co/minishlab/potion-base-32M), our best model to date. This is our recommended path if you're working with general English data.

Now that you have created the classifier, let's just train a model. The example below assumes you have the [`datasets`](https://github.com/huggingface/datasets) library installed.

```python
import numpy as np
from datasets import load_dataset

# Load the subj dataset
ds = load_dataset("setfit/subj")
train = ds["train"]
test = ds["test"]

s = perf_counter()
classifier = classifier.fit(train["text"], train["label"])

print(f"Training took {int(perf_counter() - s)} seconds.")
# Training took 81 seconds
classification_report = classifier.evaluate(ds["test"]["text"], ds["test"]["label"])
print(classification_report)
# Achieved 91.0 test accuracy
```

As you can see, we got a pretty nice 91% accuracy, with only 81 seconds of training.

The training loop is handled by [`lightning`](https://pypi.org/project/lightning/). By default the training loop splits the data into a train and validation split, with 90% of the data being used for training and 10% for validation. By default, it runs with early stopping on the validation set accuracy, with a patience of 5.

Note that this model is as fast as you're used to from us:

```python
from time import perf_counter

s = perf_counter()
classifier.predict(test["text"])
print(f"Took {int((perf_counter() - s) * 1000)} milliseconds for {len(test)} instances on CPU.")
# Took 67 milliseconds for 2000 instances on CPU.
```

## Multi-label classification

Multi-label classification is supported out of the box. Just pass a list of lists to the `fit` function (e.g. `[[label1, label2], [label1, label3]]`), and a multi-label classifier will be trained. For example, the following code trains a multi-label classifier on the [go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset:

```python
from datasets import load_dataset
from model2vec.train import StaticModelForClassification

# Initialize a classifier from a pre-trained model
classifier = StaticModelForClassification.from_pretrained(model_name="minishlab/potion-base-32M")

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
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

classification_report = classifier.evaluate(ds["test"]["text"], ds["test"]["labels"], threshold=0.3)
print(classification_report)
# Accuracy: 0.410
# Precision: 0.527
# Recall: 0.410
# F1: 0.439
```

The scores are competitive with the popular [roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) model, while our model is orders of magnitude faster.

# Persistence

You can turn a classifier into a scikit-learn compatible pipeline, as follows:

```python
pipeline = classifier.to_pipeline()
```

This pipeline object can be persisted using standard pickle-based methods, such as [joblib](https://joblib.readthedocs.io/en/stable/). This makes it easy to use your model in inferene pipelines (no installing torch!), although `joblib` and `pickle` should not be used to share models outside of your organization.

If you want to persist your pipeline to the Hugging Face hub, you can use our built-in functions:

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


# Bring your own architecture

Our training architecture is set up to be extensible, with each task having a specific class. Right now, we only offer `StaticModelForClassification`, but in the future we'll also offer regression, etc.

The core functionality of the `StaticModelForClassification` is contained in a couple of functions:

* `construct_head`: This function constructs the classifier on top of the staticmodel. For example, if you want to create a model that has LayerNorm, just subclass, and replace this function. This should be the main function to update if you want to change model behavior.
* `train_test_split`: governs the train test split before classification.
* `prepare_dataset`: Selects the `torch.Dataset` that will be used in the `Dataloader` during training.
* `_encode`: The encoding function used in the model.
* `fit`: contains all the lightning-related fitting logic.

The training of the model is done in a `lighting.LightningModule`, which can be modified but is very basic.

# Results

We ran extensive benchmarks where we compared our model to several well known architectures. The results can be found in the [training results](https://github.com/MinishLab/model2vec/tree/main/results#training-results) documentation.
