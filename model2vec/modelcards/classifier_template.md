---
{{ card_data }}
---

# {{ model_name }} Model Card

This [Model2Vec](https://github.com/MinishLab/model2vec) model is a fine-tuned version of {% if base_model %}the [{{ base_model }}](https://huggingface.co/{{ base_model }}){% else %}a{% endif %} Model2Vec model. It also includes a classifier head on top.

## Installation

Install model2vec using pip:
```
pip install model2vec[inference]
```

## Usage
Load this model using the `from_pretrained` method:
```python
from model2vec.inference import StaticModelPipeline

# Load a pretrained Model2Vec model
model = StaticModelPipeline.from_pretrained("{{ model_name }}")

# Predict labels
predicted = model.predict(["Example sentence"])
```

## Additional Resources

- [All Model2Vec models on the hub](https://huggingface.co/models?library=model2vec)
- [Model2Vec Repo](https://github.com/MinishLab/model2vec)
- [Model2Vec Results](https://github.com/MinishLab/model2vec?tab=readme-ov-file#results)
- [Model2Vec Tutorials](https://github.com/MinishLab/model2vec/tree/main/tutorials)

## Library Authors

Model2Vec was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of [Stephan Tulkens](https://github.com/stephantul) and [Thomas van Dongen](https://github.com/Pringled).

## Citation

Please cite the [Model2Vec repository](https://github.com/MinishLab/model2vec) if you use this model in your work.
```
@software{minishlab2024model2vec,
  authors = {Stephan Tulkens, Thomas van Dongen},
  title = {Model2Vec: Turn any Sentence Transformer into a Small Fast Model},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec},
}
```
