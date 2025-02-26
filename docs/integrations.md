
# Integrations

Model2Vec can be used in a variety of downstream libraries. This document provides examples of how to use Model2Vec in some of these libraries.

## Table of Contents
- [Sentence Transformers](#sentence-transformers)
- [LangChain](#langchain)
- [Txtai](#txtai)
- [Chonkie](#chonkie)
- [Transformers.js](#transformersjs)

## Sentence Transformers

Model2Vec can be used directly in [Sentence Transformers](https://github.com/UKPLab/sentence-transformers):

The following code snippet shows how to load a Model2Vec model into a Sentence Transformer model:
```python
from sentence_transformers import SentenceTransformer

# Load a Model2Vec model from the Hub
model = SentenceTransformer("minishlab/potion-base-8M")
# Make embeddings
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


## LangChain

Model2Vec can be used in [LangChain](https://github.com/langchain-ai/langchain) using the `langchain-community` package. For more information, see the [LangChain Model2Vec docs](https://python.langchain.com/docs/integrations/text_embedding/model2vec/). The following code snippet shows how to use Model2Vec in LangChain after installing the `langchain-community` package with `pip install langchain-community`:

```python
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Initialize a Model2Vec embedder
embedder = Model2vecEmbeddings("minishlab/potion-base-8M")

# Create some example texts
texts = [
    "Enduring Stew",
    "Hearty Elixir",
    "Mighty Mushroom Risotto",
    "Spicy Meat Skewer",
    "Fruit Salad",
]

# Embed the texts
embeddings = embedder.embed_documents(texts)

# Or, create a vector store and query it
documents = [Document(page_content=text) for text in texts]
vector_store = FAISS.from_documents(documents, embedder)
query = "Risotto"
query_vector = embedder.embed_query(query)
retrieved_docs = vector_store.similarity_search_by_vector(query_vector, k=1)
```

## Txtai

Model2Vec can be used in [txtai](https://github.com/neuml/txtai) for text embeddings, nearest-neighbors search, and any of the other functionalities that txtai offers. The following code snippet shows how to use Model2Vec in txtai after installing the `txtai` package (including the `vectors` dependency) with `pip install txtai[vectors]`:

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

## Chonkie

Model2Vec is the default model for semantic chunking in [Chonkie](https://github.com/bhavnicksm/chonkie). To use Model2Vec for semantic chunking in Chonkie, simply install Chonkie with `pip install chonkie[semantic]` and use one of the `potion` models in the `SemanticChunker` class. The following code snippet shows how to use Model2Vec in Chonkie:

```python
from chonkie import SDPMChunker

# Create some example text to chunk
text = "It's dangerous to go alone! Take this."

# Initialize the SemanticChunker with a potion model
chunker = SDPMChunker(
    embedding_model="minishlab/potion-base-8M",
    similarity_threshold=0.3
)

# Chunk the text
chunks = chunker.chunk(text)
```

## Transformers.js

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
