---
{{ card_data }}
---

# {{ model_name }} Model Card

This is an ONNX export of {% if base_model %}the [{{ base_model }}](https://huggingface.co/{{ base_model }}) [Model2Vec](https://github.com/MinishLab/model2vec) model{% else %}a [Model2Vec](https://github.com/MinishLab/model2vec) model{% endif %}, produced with the [ONNX](https://onnx.ai/) runtime. [Model2Vec](https://github.com/MinishLab/model2vec) models use static embeddings, allowing text embeddings to be computed orders of magnitude faster on both GPU and CPU. This ONNX export lets you run the model with `onnxruntime` or `transformers.js`, without depending on the `model2vec` package.

## Usage

### Using ONNX Runtime

```python
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{{ model_name }}")
session = ort.InferenceSession("model.onnx")

encodings = tokenizer(["Example sentence"], padding=True, return_tensors="np")
embeddings = session.run(None, dict(encodings))[0]
```

### Using the original Model2Vec model

If you don't need the ONNX runtime, you can load the original model with the [Model2Vec library](https://github.com/MinishLab/model2vec) instead:
```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("{{ base_model }}")
embeddings = model.encode(["Example sentence"])
```

## Additional Resources

- [Model2Vec Repo](https://github.com/MinishLab/model2vec)
- [Model2Vec Base Models](https://huggingface.co/collections/minishlab/model2vec-base-models-66fd9dd9b7c3b3c0f25ca90e)
- [Model2Vec Results](https://github.com/MinishLab/model2vec/tree/main/results)
- [Model2Vec Docs](https://minish.ai/packages/model2vec/introduction)

## Library Authors

Model2Vec was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of [Stephan Tulkens](https://github.com/stephantul) and [Thomas van Dongen](https://github.com/Pringled).

## Citation

Please cite the [Model2Vec repository](https://github.com/MinishLab/model2vec) if you use this model in your work.
```
@software{minishlab2024model2vec,
  author       = {Stephan Tulkens and {van Dongen}, Thomas},
  title        = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17270888},
  url          = {https://github.com/MinishLab/model2vec},
  license      = {MIT}
}
```
