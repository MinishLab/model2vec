from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import mteb
import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from model2vec.distill import distill_from_model
from model2vec.model import StaticModel

BASE_MODELS: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "baai/bge-base-en-v1.5",
    "intfloat/multilingual-e5-base",
    "google/embeddinggemma-300m",
    "Alibaba-NLP/gte-modernbert-base",
)

BASELINE_DIR = Path(__file__).parent / "data"

STS_TASKS: tuple[str, ...] = (
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
)

_NOVEL_VOCABULARY = ["zibblorptron", "quixnorfle", "blorptastic", "flimzycrag"]

CONFIGS: dict[str, dict[str, Any]] = {
    "subword": {"pca_dims": 256, "quantize_to": "float32"},
    "custom_vocab": {"vocabulary": _NOVEL_VOCABULARY, "pca_dims": 32, "quantize_to": "float32"},
}


def baseline_path_for(model_name: str) -> Path:
    """The JSON baseline file for a given short model name (a key of `BASE_MODELS`)."""
    safe_model_name = model_name.replace("/", "___")
    return BASELINE_DIR / f"{safe_model_name}_baseline.json"


def load_base_model_and_tokenizer(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Download a base sentence-transformer and its tokenizer once, for reuse across distillations.

    :param model_name: The HuggingFace model id to download, e.g. `BASE_MODELS["minilm"]`.
    :return: The loaded model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name)
    tokenizer = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(model_name, use_fast=True))
    return model, tokenizer


def distill_all(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast) -> dict[str, StaticModel]:
    """Distill every configured variant from the same base model and tokenizer."""
    return {name: distill_from_model(model=model, tokenizer=tokenizer, **kwargs) for name, kwargs in CONFIGS.items()}


class _MTEBEncoder(AbsEncoder):
    """Adapts a `StaticModel` to MTEB's `AbsEncoder` protocol."""

    def __init__(self, model: StaticModel) -> None:
        self.model = model
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={"name": "model2vec-distilled", "revision": "local"})

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        texts = [text for batch in inputs for text in batch["text"]]
        return self.model.encode(texts)


def compute_mteb_sts_scores(model: StaticModel) -> dict[str, float]:
    """Run the STS subset of MTEB against a distilled model.

    :param model: The distilled StaticModel to evaluate.
    :return: A dict mapping MTEB STS task name to its main score.
    """
    tasks = mteb.get_tasks(tasks=list(STS_TASKS))
    results = mteb.evaluate(_MTEBEncoder(model), tasks, cache=None, show_progress_bar=False)  # type: ignore[arg-type]
    return {result.task_name: round(float(result.get_score()), 6) for result in results.task_results}


def compute_metrics(model: StaticModel) -> dict[str, Any]:
    """Compute a JSON-serializable snapshot of a distilled model's key properties.

    :param model: The distilled StaticModel to summarize.
    :return: A dict with vocab size, embedding shape/rank/distribution, token order, and MTEB STS
        scores. Used both to write and to check the regression baseline.
    """
    embedding = model.embedding.astype(np.float64)
    tokens = list(model.tokens)
    token_order_hash = hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()

    return {
        "full_vocab_size": len(tokens),
        "embedding_rows": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]),
        "embedding_rank": int(np.linalg.matrix_rank(embedding)),
        "embedding_mean": round(float(embedding.mean()), 6),
        "embedding_std": round(float(embedding.std()), 6),
        "token_order_hash": token_order_hash,
        "first_tokens": tokens[:10],
        "last_tokens": tokens[-10:],
        "mteb_sts_scores": compute_mteb_sts_scores(model),
    }
