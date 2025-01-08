from model2vec.utils import get_package_extras, importable

_REQUIRED_EXTRA = "train"

for extra_dependency in get_package_extras("model2vec", _REQUIRED_EXTRA):
    importable(extra_dependency, _REQUIRED_EXTRA)

from model2vec.train.classifier import StaticModelForClassification

__all__ = ["StaticModelForClassification"]
