# Knowledge Distillation

# Installation

For knowledge distillation, make sure you install the knowledge_distill extra.

```
pip install model2vec[knowledge_distill]
```


# Quickstart

Create features:

```bash
python3 -m tokenlearn.featurize --model-name "baai/bge-base-en-v1.5" --output-dir "data/c4_features"
```

Knowledge distillation:

```python
from model2vec.knowledge_distill import KnowledgeDistillationModel
