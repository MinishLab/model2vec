# Training

Aside from [distillation](../../README.md#distillation), `model2vec` also supports training simple classifiers on top of static models, using [pytorch](https://pytorch.org/) and [lightning](https://lightning.ai/).

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
classifier = StaticModelForClassification.from_static_model(distilled_model)

# From a pre-trained model: potion is the default
classifier = StaticModelForClassification.from_pretrained(model_name="minishlab/potion-base-8m")
```

This creates a very simple classifier: a StaticModel with a single 512-unit hidden layer on top. You can adjust the number of hidden layers and the number units through some parameters on both functions. Note that the default for `from_pretrained` is [potion-base-8m](https://huggingface.co/minishlab/potion-base-8M), our best model to date. This is our recommended path if you're working with general English data.

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

predicted = classifier.predict(test["text"])
print(f"Training took {int(perf_counter() - s)} seconds.")
# Training took 81 seconds
accuracy = np.mean([x == y for x, y in zip(predicted, test["label"])]) * 100
print(f"Achieved {accuracy} test accuracy")
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

# Persistence

You can turn a classifier into a scikit-learn compatible pipeline, as follows:

```python
pipeline = classifier.to_pipeline()
```

This pipeline object can be persisted using standard pickle-based methods, such as [joblib](https://joblib.readthedocs.io/en/stable/). This makes it easy to use your model in inferene pipelines (no installing torch!).

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

# Results

The main results are detailed in our training blogpost, but we'll do a comparison with vanilla model2vec here. In a vanilla model2vec classifier, you just put a scikit-learn `LogisticRegressionCV` on top of the model encoder. In contrast, training a `StaticModelForClassification` fine-tunes the full model, including the `StaticModel` weights.

We use 14 classification datasets, using 1000 examples from the train set, and the full test set. No parameters were tuned on any validation set. All datasets were taken from the [Setfit organization on Hugging Face](https://huggingface.co/datasets/SetFit).

| dataset name               |   logistic regression head   |  full finetune |
|:---------------------------|-----------:|---------------:|
| 20_newgroups               |   0.545312 |       0.555459 |
| ade                        |   0.715725 |  0.740307 |
| ag_news                    |   0.860154 |  0.858304 |
| amazon_counterfactual      |   0.637754 |  0.744288 |
| bbc                        |   0.955719 |  0.965018 |
| emotion                    |   0.516267 |  0.586328 |
| enron_spam                 |   0.951975 |  0.964994 |
| hatespeech_offensive       |   0.543758 |  0.592587 |
| imdb                       |   0.839002 |  0.846198 |
| massive_scenario           |   0.797779 |  0.822825 |
| senteval_cr                |   0.743436 |  0.745863 |
| sst5                       |   0.290249 |  0.363071 |
| student                    |   0.806069 |  0.837581 |
| subj                       |   0.878394 |  0.88941  |
| tweet_sentiment_extraction |   0.638664 |  0.632009 |

|                |   logreg   |  full finetune |
|:---------------------------|-----------:|---------------:|
| average                    |   0.714    |    0.742       |

As you can see, full fine-tuning brings modest performance improvements in some cases, but very large ones in other cases, leading to a pretty large increase in average score. Our advice is to test both if you can use `potion-base-8m`, and to use full fine-tuning if you are starting from another base model.

# Bring your own architecture

Our training architecture is set up to be extensible, with each task having a specific class. Right now, we only offer `StaticModelForClassification`, but in the future we'll also offer regression, etc.

The core functionality of the `StaticModelForClassification` is contained in a couple of functions:

* `construct_head`: This function constructs the classifier on top of the staticmodel. For example, if you want to create a model that has LayerNorm, just subclass, and replace this function. This should be the main function to update if you want to change model behavior.
* `train_test_split`: governs the train test split before classification.
* `prepare_dataset`: Selects the `torch.Dataset` that will be used in the `Dataloader` during training.
* `_encode`: The encoding function used in the model.
* `fit`: contains all the lightning-related fitting logic.

The training of the model is done in a `lighting.LightningModule`, which can be modified but is very basic.
