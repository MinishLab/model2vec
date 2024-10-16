from model2vec.utils import get_package_extras, importable

_REQUIRED_EXTRA = "distill"

for extra_dependency in get_package_extras("model2vec", _REQUIRED_EXTRA):
    importable(extra_dependency, _REQUIRED_EXTRA)

from model2vec.distill.distillation import distill, distill_from_model

__all__ = ["distill", "distill_from_model"]
