from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
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

_NOVEL_VOCABULARY = ["zibblorptron", "quixnorfle", "blorptastic", "flimzycrag"]

CONFIGS: dict[str, dict[str, Any]] = {
    "subword": {"pca_dims": 256, "quantize_to": "float32"},
    "custom_vocab": {"vocabulary": _NOVEL_VOCABULARY, "pca_dims": 32, "quantize_to": "float32"},
}

SEMANTIC_TRIPLES = [
    ("king", "queen", "bicycle"),
    ("dog", "puppy", "astronomy"),
    ("happy", "joyful", "concrete"),
]


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


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def compute_similarity_scores(model: StaticModel) -> dict[str, float]:
    """Compute cosine similarity scores for the semantic sanity pairs."""
    scores: dict[str, float] = {}
    for word_a, word_b, unrelated in SEMANTIC_TRIPLES:
        vectors = model.encode([word_a, word_b, unrelated])
        scores[f"{word_a}~{word_b}"] = round(_cosine_similarity(vectors[0], vectors[1]), 6)
        scores[f"{word_a}~{unrelated}"] = round(_cosine_similarity(vectors[0], vectors[2]), 6)
    return scores


def compute_metrics(model: StaticModel) -> dict[str, Any]:
    """Compute a JSON-serializable snapshot of a distilled model's key properties.

    :param model: The distilled StaticModel to summarize.
    :return: A dict with vocab size, embedding shape/rank/distribution, token order, and semantic
        similarity scores. Used both to write and to check the regression baseline.
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
        "similarity_scores": compute_similarity_scores(model),
    }
