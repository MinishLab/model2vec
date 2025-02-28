from model2vec.utils import get_package_extras, importable

_REQUIRED_EXTRA = "inference"

for extra_dependency in get_package_extras("model2vec", _REQUIRED_EXTRA):
    importable(extra_dependency, _REQUIRED_EXTRA)

from model2vec.inference.model import StaticModelPipeline, evaluate_single_or_multi_label

__all__ = ["StaticModelPipeline", "evaluate_single_or_multi_label"]
