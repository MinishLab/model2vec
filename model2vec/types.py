from __future__ import annotations

from typing import Any, TypedDict


class _UnsetType:
    pass


_UNSET = _UnsetType()


class StaticModelConfig(TypedDict, total=False):
    """The metadata config stored alongside a model2vec model, e.g. in `config.json`."""

    normalize: bool
    max_length: int | None
    model_type: str
    architectures: list[str]
    tokenizer_name: str
    apply_pca: int | float | str | None
    sif_coefficient: float | None
    hidden_dim: int
    seq_length: int
    pooling: str
    embedding_dtype: str
    vocabulary_quantization: int
    head_config: dict[str, Any]
