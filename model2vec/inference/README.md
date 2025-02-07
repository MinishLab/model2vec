# Inference

This subpackage mainly contains helper functions for inference with trained models that have been exported to `scikit-learn` compatible pipelines.

If you're looking for information on how to train a model, see [here](../train/README.md).

# Usage

Let's assume you're using our [potion-edu classifier](https://huggingface.co/minishlab/potion-8m-edu-classifier).

```python
from model2vec.inference import StaticModelPipeline

classifier = StaticModelPipeline.from_pretrained("minishlab/potion-8m-edu-classifier")
label = classifier.predict("Attitudes towards cattle in the Alps: a study in letting go.")
```

This should just work.
