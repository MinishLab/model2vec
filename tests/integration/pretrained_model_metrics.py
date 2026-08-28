from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from model2vec.model import StaticModel
from tests.integration.distill_metrics import STS_TASKS, compute_mteb_sts_scores

PRETRAINED_MODELS: tuple[str, ...] = (
    "minishlab/potion-base-8m",
    "minishlab/potion-base-32m",
    "minishlab/potion-retrieval-32m",
    "minishlab/potion-multilingual-128m",
)

BASELINE_DIR = Path(__file__).parent / "data" / "pretrained"

_SPEED_SENTENCES: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog.",
    "Paris is the capital of France, and it sits on the river Seine.",
    "I would like to order a large pizza with extra cheese and mushrooms.",
    "Machine learning models can be trained on very large text datasets.",
    "The weather today is sunny with a light breeze coming off the sea.",
    "She sells seashells by the seashore every single summer morning.",
    "Static embedding models trade a little accuracy for a lot of speed.",
    "hi",
)
_SPEED_CORPUS: list[str] = [_SPEED_SENTENCES[i % len(_SPEED_SENTENCES)] for i in range(2048)]
_SPEED_RUNS = 3


def baseline_path_for(model_name: str) -> Path:
    """The JSON baseline file for a pretrained model id, e.g. `minishlab/potion-base-8m`."""
    safe_model_name = model_name.replace("/", "___")
    return BASELINE_DIR / f"{safe_model_name}_baseline.json"


def load_static_model(model_name: str) -> StaticModel:
    """Download a published `StaticModel` from the Hugging Face hub."""
    return StaticModel.from_pretrained(model_name)


def measure_encoding_speed(model: StaticModel) -> dict[str, float]:
    """Encode a fixed corpus a few times and report the best throughput.

    :param model: The model to benchmark.
    :return: Best-of-N sentences/second and tokens/second over `_SPEED_CORPUS`.
    """
    model.encode(_SPEED_CORPUS[:64], use_multiprocessing=False)

    n_tokens = sum(len(ids) for ids in model.tokenize(_SPEED_CORPUS))
    best_seconds = float("inf")
    for _ in range(_SPEED_RUNS):
        start = time.perf_counter()
        model.encode(_SPEED_CORPUS, use_multiprocessing=False)
        best_seconds = min(best_seconds, time.perf_counter() - start)

    return {
        "sentences_per_second": round(len(_SPEED_CORPUS) / best_seconds, 2),
        "tokens_per_second": round(n_tokens / best_seconds, 2),
    }


def compute_metrics(model: StaticModel) -> dict[str, Any]:
    """Compute a JSON-serializable snapshot of a published model's properties.

    :param model: The loaded StaticModel to summarize.
    :return: A dict with every loaded attribute, embedding stats, MTEB STS scores and encoding speed.
    """
    embedding = model.embedding.astype(np.float64)
    tokens = list(model.tokens)
    token_order_hash = hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()

    return {
        "full_vocab_size": len(tokens),
        "embedding_rows": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]),
        "embedding_dtype": model.embedding_dtype,
        "embedding_rank": int(np.linalg.matrix_rank(embedding)),
        "embedding_mean": round(float(embedding.mean()), 6),
        "embedding_std": round(float(embedding.std()), 6),
        "embedding_row_norm_mean": round(float(np.linalg.norm(embedding, axis=1).mean()), 6),
        "token_order_hash": token_order_hash,
        "first_tokens": tokens[:10],
        "last_tokens": tokens[-10:],
        "median_token_length": int(model.median_token_length),
        "unk_token_id": model.unk_token_id,
        "normalize": bool(model.normalize),
        "base_model_name": model.base_model_name,
        "language": model.language,
        "vocabulary_quantization": model.vocabulary_quantization,
        "has_weights": model.weights is not None,
        "has_token_mapping": model.token_mapping is not None,
        "tokenizer_type": type(model.tokenizer.model).__name__,
        "config": model.config,
        "mteb_sts_scores": compute_mteb_sts_scores(model),
        "encoding_speed": measure_encoding_speed(model),
    }


__all__ = [
    "BASELINE_DIR",
    "PRETRAINED_MODELS",
    "STS_TASKS",
    "baseline_path_for",
    "compute_metrics",
    "load_static_model",
    "measure_encoding_speed",
]
