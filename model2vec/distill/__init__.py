from importlib import import_module

try:
    import_module("torch")
except ImportError:
    raise ImportError(
        "torch, scikit-learn and transformers are required for distillation. Please reinstall model2vec with the 'distill' extra."
    )

from model2vec.distill.distillation import distill, distill_from_model

__all__ = ["distill", "distill_from_model"]
