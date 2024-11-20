
<div align="center">
    <picture>
      <img width="35%" alt="Model2Vec logo" src="assets/images/logo_v2.png">
    </picture>
  </a>
</div>

<div align="center">
  <h2>The Fastest State-of-the-Art Static Embeddings in the World</h2>
</div>

<div align="center">
  <h2>
    <a href="https://huggingface.co/minishlab"><strong>🤗 Models</strong></a> |
    <a href="https://github.com/MinishLab/model2vec/tree/main/tutorials"><strong>📚 Tutorials</strong></a> |
    <a href="https://minishlab.github.io/"><strong>🌐 Website</strong></a> |
    <a href="https://github.com/MinishLab/model2vec/blob/main/results/README.md"><strong>🏆 Results</strong></a>
  </h2>
</div>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/model2vec/"><img src="https://img.shields.io/pypi/v/model2vec?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://pypi.org/project/model2vec/"><img src="https://img.shields.io/pypi/pyversions/model2vec" alt="Supported Python versions"></a>
    <a href="https://pepy.tech/project/model2vec">
    <img src="https://static.pepy.tech/badge/model2vec" alt="Downloads">
    </a>
    <a href="https://app.codecov.io/gh/MinishLab/model2vec">
    <img src="https://codecov.io/gh/MinishLab/model2vec/graph/badge.svg?token=21TWJ6B5ET" alt="Codecov">
    </a>
  </a>
    <a href="https://github.com/MinishLab/model2vec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT"></a>
  </h2>
</div>

<div align="center">
    <img src="assets/images/model2vec_model_diagram_transparant_dark.png#gh-dark-mode-only" width="90%">
    <img src="assets/images/model2vec_model_diagram_transparant_light.png#gh-light-mode-only" width="90%">
</div>

Model2Vec is a technique to turn any sentence transformer into a really small static model, reducing model size by 15x and making the models up to 500x faster, with a small drop in performance. Our [best model](https://huggingface.co/minishlab/potion-base-8M) is the most performant static embedding model in the world. See our results [here](results/README.md), or dive in to see how it works.


## Updates & Announcements

- **30/10/2024**: We released three new models: [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M), [potion-base-4M](https://huggingface.co/minishlab/potion-base-4M), and [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M). These models are trained using [Tokenlearn](https://github.com/MinishLab/tokenlearn). Find out more in our [blog post](https://minishlab.github.io/tokenlearn_blogpost/). NOTE: for users of any of our old English M2V models, we recommend switching to these new models as they [perform better on all tasks](https://github.com/MinishLab/model2vec/tree/main/results).

## Table of Contents
- [Quickstart](#quickstart)
- [Main Features](#main-features)
- [What is Model2Vec?](#what-is-model2vec)
- [Usage](#usage)
    - [Inference](#inference)
    - [Distillation](#distillation)
    - [Evaluation](#evaluation)
- [Integrations](#integrations)
- [Model List](#model-list)
- [Results](#results)

## Quickstart

Install the package with:

```bash
pip install model2vec
```

This will install the base inference package, which only depends on `numpy` and a few other minor dependencies. If you want to distill your own models, you can install the distillation extras with:

```bash
pip install model2vec[distill]
```

The easiest way to get started with Model2Vec is to load one of our [flagship models from the HuggingFace hub](https://huggingface.co/collections/minishlab/potion-6721e0abd4ea41881417f062). These models are pre-trained and ready to use. The following code snippet shows how to load a model and make embeddings:
```python
from model2vec import StaticModel

# Load a model from the HuggingFace hub (in this case the potion-base-8M model)
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Make embeddings
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])

# Make sequences of token embeddings
token_embeddings = model.encode_as_sequence(["It's dangerous to go alone!", "It's a secret to everybody."])
```

And that's it. You can use the model to classify texts, to cluster, or to build a RAG system.

Instead of using one of our models, you can also distill your own Model2Vec model from a Sentence Transformer model. The following code snippet shows how to distill a model:
```python
from model2vec.distill import distill

# Distill a Sentence Transformer model, in this case the BAAI/bge-base-en-v1.5 model
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5", pca_dims=256)

# Save the model
m2v_model.save_pretrained("m2v_model")
```

Distillation is really fast and only takes 30 seconds on CPU. Best of all, distillation requires no training data.

For advanced usage, such as using Model2Vec in the [Sentence Transformers library](https://github.com/UKPLab/sentence-transformers), please refer to the [Usage](#usage) sections.


## Main Features

- **State-of-the-Art Performance**: Model2Vec models outperform any other static embeddings (such as GLoVe and BPEmb) by a large margin, as can be seen in our [results](results/README.md).
- **Small**: Model2Vec reduces the size of a Sentence Transformer model by a factor of 15, from 120M params, down to 7.5M (30 MB on disk, making it the smallest model on [MTEB](https://huggingface.co/spaces/mteb/leaderboard)!).
- **Lightweight Dependencies**: the base package's only major dependency is `numpy`.
- **Lightning-fast Inference**: up to 500 times faster on CPU than the original model. Go green or go home.
- **Fast, Dataset-free Distillation**: distill your own model in 30 seconds on a CPU, without a dataset. All you need is a model and (optionally) a custom vocabulary.
- **Integrated into Sentence Transformers and txtai**: Model2Vec can be used directly in [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and [txtai](https://github.com/neuml/txtai).
- **Tightly integrated with HuggingFace hub**: easily share and load models from the HuggingFace hub, using the familiar `from_pretrained` and `push_to_hub`. Our own models can be found [here](https://huggingface.co/minishlab). Feel free to share your own.

## What is Model2Vec?

Model2vec creates a small, fast, and powerful model that outperforms other static embedding models by a large margin on all tasks we could find, while being much faster to create than traditional static embedding models such as GloVe. Like BPEmb, it can create subword embeddings, but with much better performance. Distillation doesn't need _any_ data, just a vocabulary and a model.

The base model2vec technique works by passing a vocabulary through a sentence transformer model, then reducing the dimensionality of the resulting embeddings using PCA, and finally weighting the embeddings using zipf weighting. During inference, we simply take the mean of all token embeddings occurring in a sentence.

Our [potion models](https://huggingface.co/collections/minishlab/potion-6721e0abd4ea41881417f062) are pre-trained using [tokenlearn](https://github.com/MinishLab/tokenlearn), a technique to pre-train model2vec distillation models. These models are created with the following steps:
- **Distillation**: We distill a Model2Vec model from a Sentence Transformer model, using the method described above.
- **Sentence Transformer inference**: We use the Sentence Transformer model to create mean embeddings for a large number of texts from a corpus.
- **Training**: We train a model to minimize the cosine distance between the mean embeddings generated by the Sentence Transformer model and the mean embeddings generated by the Model2Vec model.
- **Post-training re-regularization**: We re-regularize the trained emebeddings by first performing PCA, and then weighting the embeddings using `smooth inverse frequency (SIF)` weighting using the following formula: `w = 1e-3 / (1e-3 + proba)`. Here, `proba` is the probability of the token in the corpus we used for training.


For a much more extensive deepdive, please refer to our [Model2Vec blog post](https://huggingface.co/blog/Pringled/model2vec) and our [Tokenlearn blog post](https://minishlab.github.io/tokenlearn_blogpost/).

## Usage

### Inference

<details>
<summary>  Inference with a pretrained model </summary>
<br>

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
</details>


<details>
<summary>  Inference with the Sentence Transformers library </summary>
<br>

The following code snippet shows how to use a Model2Vec model in the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) library. This is useful if you want to use the model in a Sentence Transformers pipeline.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

# Initialize a StaticEmbedding module
static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
```

</details>

### Distillation

<details>
<summary>  Distilling from a Sentence Transformer </summary>
<br>

The following code can be used to distill a model from a Sentence Transformer. As mentioned above, this leads to really small model that might be less performant.
```python
from model2vec.distill import distill

# Distill a Sentence Transformer model
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5", pca_dims=256)

# Save the model
m2v_model.save_pretrained("m2v_model")

```
</details>

<details>
<summary>  Distilling from a loaded model </summary>
<br>

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

</details>

<details>
<summary>  Distilling with the Sentence Transformers library </summary>
<br>

The following code snippet shows how to distill a model using the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) library. This is useful if you want to use the model in a Sentence Transformers pipeline.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

static_embedding = StaticEmbedding.from_distillation("BAAI/bge-base-en-v1.5", device="cpu", pca_dims=256)
model = SentenceTransformer(modules=[static_embedding])
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
```

</details>


<details>
<summary>  Distilling with a custom vocabulary </summary>
<br>

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

</details>


### Evaluation


<details>
<summary>  Installation </summary>
<br>

Our models can be evaluated using our [evaluation package](https://github.com/MinishLab/evaluation). Install the evaluation package with:

```bash
pip install git+https://github.com/MinishLab/evaluation.git@main
```
</details>

<details>
  <summary>  Evaluation Code </summary>
<br>

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
</details>

## Integrations
<details>
<summary>  Sentence Transformers </summary>
<br>

Model2Vec can be used directly in [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) using the `StaticEmbedding` module.

The following code snippet shows how to load a Model2Vec model into a Sentence Transformer model:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

# Initialize a StaticEmbedding module
static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
```

The following code snippet shows how to distill a model directly into a Sentence Transformer model:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

static_embedding = StaticEmbedding.from_distillation("BAAI/bge-base-en-v1.5", device="cpu", pca_dims=256)
model = SentenceTransformer(modules=[static_embedding])
embeddings = model.encode(["It's dangerous to go alone!", "It's a secret to everybody."])
```

For more documentation, please refer to the [Sentence Transformers documentation](https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.StaticEmbedding).

</details>


<details>
<summary>  Txtai </summary>
<br>

Model2Vec can be used in [txtai](https://github.com/neuml/txtai) for text embeddings, nearest-neighbors search, and any of the other functionalities that txtai offers. The following code snippet shows how to use Model2Vec in txtai:

```python
from txtai import Embeddings

# Load a model2vec model
embeddings = Embeddings(path="minishlab/potion-base-8M", method="model2vec", backend="numpy")

# Create some example texts
texts = ["Enduring Stew", "Hearty Elixir", "Mighty Mushroom Risotto", "Spicy Meat Skewer", "Chilly Fruit Salad"]

# Create embeddings for downstream tasks
vectors = embeddings.batchtransform(texts)

# Or create a nearest-neighbors index and search it
embeddings.index(texts)
result = embeddings.search("Risotto", 1)
```

</details>

<details>
<summary>  Transformers.js </summary>

<br>

To use a Model2Vec model in [transformers.js](https://github.com/huggingface/transformers.js), the following code snippet can be used as a starting point:

```javascript
import { AutoModel, AutoTokenizer, Tensor } from '@huggingface/transformers';

const modelName = 'minishlab/potion-base-8M';

const modelConfig = {
    config: { model_type: 'model2vec' },
    dtype: 'fp32',
    revision: 'refs/pr/1'
};
const tokenizerConfig = {
    revision: 'refs/pr/2'
};

const model = await AutoModel.from_pretrained(modelName, modelConfig);
const tokenizer = await AutoTokenizer.from_pretrained(modelName, tokenizerConfig);

const texts = ['hello', 'hello world'];
const { input_ids } = await tokenizer(texts, { add_special_tokens: false, return_tensor: false });

const cumsum = arr => arr.reduce((acc, num, i) => [...acc, num + (acc[i - 1] || 0)], []);
const offsets = [0, ...cumsum(input_ids.slice(0, -1).map(x => x.length))];

const flattened_input_ids = input_ids.flat();
const modelInputs = {
    input_ids: new Tensor('int64', flattened_input_ids, [flattened_input_ids.length]),
    offsets: new Tensor('int64', offsets, [offsets.length])
};

const { embeddings } = await model(modelInputs);
console.log(embeddings.tolist()); // output matches python version
```

Note that this requires that the Model2Vec has a `model.onnx` file and several required tokenizers file. To generate these for a model that does not have them yet, the following code snippet can be used:

```bash
python scripts/export_to_onnx.py --model_path <path-to-a-model2vec-model> --save_path "<path-to-save-the-onnx-model>"
```


<br>
</details>


## Model List

We provide a number of models that can be used out of the box. These models are available on the [HuggingFace hub](https://huggingface.co/collections/minishlab/model2vec-base-models-66fd9dd9b7c3b3c0f25ca90e) and can be loaded using the `from_pretrained` method. The models are listed below.


| Model                                                                 | Language    | Vocab            | Sentence Transformer                                            | Tokenizer Type | Params  | Tokenlearn        |
|-----------------------------------------------------------------------|-------------|------------------|-----------------------------------------------------------------|----------------|---------|-------------------|
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)     | English     | Output           | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | Subword        | 7.5M    | <div align="center">✅</div> |
| [potion-base-4M](https://huggingface.co/minishlab/potion-base-4M)     | English     | Output           | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | Subword        | 3.7M    | <div align="center">✅</div> |
| [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M)     | English     | Output           | [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | Subword        | 1.8M    | <div align="center">✅</div> |
| [M2V_multilingual_output](https://huggingface.co/minishlab/M2V_multilingual_output) | Multilingual | Output           | [LaBSE](https://huggingface.co/sentence-transformers/LaBSE)      | Subword        | 471M    | <div align="center">❌</div> |



## Results

We have performed extensive experiments to evaluate the performance of Model2Vec models. The results are documented in the [results](results/README.md) folder. The results are presented in the following sections:
- [MTEB Results](results/README.md#mteb-results)
- [Ablations](results/README.md#ablations)

## License

MIT

## Citing

If you use Model2Vec in your research, please cite the following:
```bibtex
@software{minishlab2024model2vec,
  authors = {Stephan Tulkens, Thomas van Dongen},
  title = {Model2Vec: The Fastest State-of-the-Art Static Embeddings in the World},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec},
}
```
