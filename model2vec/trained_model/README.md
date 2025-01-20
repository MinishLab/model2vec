# Trained

This subpackage mainly contains helper functions for working with trained models that have been exported to `scikit-learn` compatible pipelines.

If you're looking for information on how to train a model, see [here](../train/README.md).

# Usage

Let's assume you're using our `potion-edu classifier`.

```python
from model2vec.trained_model import StaticModelPipeline

s = StaticModelPipeline.from_pretrained("minishlab/potion-edu-classifier")
label = s.predict("Attitudes towards cattle in the Alps: a study in letting go.")
```

This should just work.
