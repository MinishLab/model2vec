# Inference

This subpackage mainly contains helper functions for inference with trained classifier/projector heads, persisted as a `safetensors` file and `config.json` metadata.

If you're looking for information on how to train a model, see [here](../train/README.md).

# Usage

Let's assume you're using our [potion-edu classifier](https://huggingface.co/minishlab/potion-8m-edu-classifier).

```python
from model2vec.inference import StaticModelPipeline

classifier = StaticModelPipeline.from_pretrained("minishlab/potion-8m-edu-classifier")
label = classifier.predict("Attitudes towards cattle in the Alps: a study in letting go.")
```

This should just work.

# Migrating a legacy pipeline

Pipelines saved by older versions of model2vec store the head as a `scikit-learn`/`skops` `pipeline.skops` file instead of `head.safetensors`. `from_pretrained` still loads these automatically, falling back to the legacy format and emitting a warning. This requires `scikit-learn` and `skops` to be installed.

To upgrade a pipeline to the current format (and silence the warning), convert it with `convert_legacy_pipeline` and save the result:

```python
from model2vec.inference import convert_legacy_pipeline

pipeline = convert_legacy_pipeline("path/or/repo-id/of/legacy/pipeline")
pipeline.save_pretrained("path/to/save")
```
