# Training

Aside from [distillation](../../README.md#distillation), `model2vec` also supports training simple classifiers on top of static models, using [pytorch](https://pytorch.org/), [lightning](https://lightning.ai/) and [scikit-learn](https://scikit-learn.org/stable/index.html).

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

# Results

The main results are detailed in our training blogpost, but we'll do a comparison with vanilla model2vec here. In a vanilla model2vec classifier, you just put a scikit-learn `LogisticRegressionCV` on top of the model encoder. In contrast, training a `StaticModelForClassification` fine-tunes the full model, including the `StaticModel` weights. The Setfit model is trained on using [all-minilm-l6-v2](sentence-transformers/all-MiniLM-L6-v2) as a base model.

We use 14 classification datasets, using 1000 examples from the train set, and the full test set. No parameters were tuned on any validation set. All datasets were taken from the [Setfit organization on Hugging Face](https://huggingface.co/datasets/SetFit).

| dataset               |   model2vec + logreg |   model2vec full finetune |   setfit |
|:---------------------------|----------------------------------------------:|---------------------------------------:|-------------------------------------------------:|
| 20_newgroups               |                                         56.24 |                                  57.94 |                                            61.29 |
| ade                        |                                         79.2  |                                  79.68 |                                            83.05 |
| ag_news                    |                                         86.7  |                                  87.2  |                                            88.01 |
| amazon_counterfactual      |                                         90.96 |                                  91.93 |                                            95.51 |
| bbc                        |                                         95.8  |                                  97.21 |                                            96.6  |
| emotion                    |                                         65.57 |                                  67.11 |                                            72.86 |
| enron_spam                 |                                         96.4  |                                  96.85 |                                            97.45 |
| hatespeech_offensive       |                                         83.54 |                                  85.61 |                                            87.69 |
| imdb                       |                                         85.34 |                                  85.59 |                                            86    |
| massive_scenario           |                                         82.86 |                                  84.42 |                                            83.54 |
| senteval_cr                |                                         77.03 |                                  79.47 |                                            86.15 |
| sst5                       |                                         32.34 |                                  37.95 |                                            42.31 |
| student                    |                                         83.2  |                                  85.02 |                                            89.62 |
| subj                       |                                         89.2  |                                  89.85 |                                            93.8  |
| tweet_sentiment_extraction |                                         64.96 |                                  62.65 |                                            75.15 |

|                |   logreg   |  full finetune | setfit
|:---------------------------|-----------:|---------------:|-------:|
| average                    |   77.9    |    79.2       |   82.6 |

As you can see, full fine-tuning brings modest performance improvements in some cases, but very large ones in other cases, leading to a pretty large increase in average score. Our advice is to test both if you can use `potion-base-32m`, and to use full fine-tuning if you are starting from another base model.

The speed difference between model2vec and setfit is immense, with the full finetune being 35x faster than a setfit based on `all-minilm-l6-v2` on CPU.

|                |   logreg   |  full finetune | setfit
|:---------------------------|-----------:|---------------:|-------:|
| samples / second                    |   17925    |    24744       |   716 |


# Bring your own architecture

Our training architecture is set up to be extensible, with each task having a specific class. Right now, we only offer `StaticModelForClassification`, but in the future we'll also offer regression, etc.

The core functionality of the `StaticModelForClassification` is contained in a couple of functions:

* `construct_head`: This function constructs the classifier on top of the staticmodel. For example, if you want to create a model that has LayerNorm, just subclass, and replace this function. This should be the main function to update if you want to change model behavior.
* `train_test_split`: governs the train test split before classification.
* `prepare_dataset`: Selects the `torch.Dataset` that will be used in the `Dataloader` during training.
* `_encode`: The encoding function used in the model.
* `fit`: contains all the lightning-related fitting logic.

The training of the model is done in a `lighting.LightningModule`, which can be modified but is very basic.
