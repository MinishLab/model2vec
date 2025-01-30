"""Script to benchmark the speed of various text embedding models and generate a plot of the MTEB score vs samples per second."""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from bpemb import BPEmb
from datasets import load_dataset
from plotnine import (
    aes,
    element_line,
    geom_point,
    geom_text,
    ggplot,
    guides,
    labs,
    scale_size,
    scale_y_continuous,
    theme,
    theme_classic,
    xlim,
    ylim,
)
from sentence_transformers import SentenceTransformer

from model2vec import StaticModel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class BPEmbEmbedder:
    def __init__(self, vs: int = 50_000, dim: int = 300) -> None:
        """Initialize the BPEmbEmbedder."""
        self.bpemb_en = BPEmb(lang="en", vs=vs, dim=dim)

    def mean_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Encode a sentence to a mean embedding."""
        encoded_ids = self.bpemb_en.encode_ids(sentence)
        embeddings = self.bpemb_en.vectors[encoded_ids]
        if embeddings.size == 0:
            return np.zeros(self.bpemb_en.dim)  # Return a zero vector if no tokens are found
        return embeddings.mean(axis=0)

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """Encode a list of sentences to embeddings."""
        return np.array([self.mean_sentence_embedding(sentence.lower()) for sentence in sentences])


def make_plot(df: pd.DataFrame) -> ggplot:
    """Create a plot of the MTEB score vs samples per second."""
    df["label_y"] = (
        df["Average score"]
        + 0.5  # a constant "base" offset for all bubbles
        + 0.08 * np.sqrt(df["Params (Million)"])
    )
    plot = (
        ggplot(df, aes(x="Samples per second", y="Average score"))
        + geom_point(aes(size="Params (Million)", color="Model"))
        + geom_text(aes(y="label_y", label="Model"), color="black", size=7)
        + scale_size(range=(2, 30))
        + theme_classic()
        + labs(title="Average MTEB Score vs Samples per Second")
        + ylim(df["Average score"].min(), df["Average score"].max() + 3)
        + scale_y_continuous(breaks=range(30, 70, 5))
        + theme(
            panel_grid_major=element_line(color="lightgrey", size=0.5),
            panel_grid_minor=element_line(color="lightgrey", size=0.25),
            figure_size=(10, 6),
        )
        + xlim(0, df["Samples per second"].max() + 100)
        + guides(None)
    )
    return plot


def benchmark_model(name: str, info: list[str], texts: list[str]) -> dict[str, float | str]:
    """Benchmark a single model."""
    logger.info("Starting", name)
    if info[1] == "BPEmb":
        model = BPEmbEmbedder(vs=50_000, dim=300)  # type: ignore
    elif info[1] == "ST":
        model = SentenceTransformer(info[0], device="cpu")  # type: ignore
    else:
        model = StaticModel.from_pretrained(info[0])  # type: ignore

    start = perf_counter()
    if info[1] == "M2V":
        # If the model is a model2vec model, disable multiprocessing for a fair comparison
        model.encode(texts, use_multiprocessing=False)
    else:
        model.encode(texts)

    total_time = perf_counter() - start
    docs_per_second = len(texts) / total_time

    logger.info(f"{name}: {docs_per_second} docs per second")
    logger.info(f"Total time: {total_time}")

    return {"docs_per_second": docs_per_second, "total_time": total_time}


def main(save_path: str, n_texts: int) -> None:
    """Benchmark text embedding models and generate a plot."""
    # Define the models to benchmark
    models: dict[str, list[str]] = {
        "BPEmb-50k-300d": ["", "BPEmb"],
        "all-MiniLM-L6-v2": ["sentence-transformers/all-MiniLM-L6-v2", "ST"],
        "bge-base-en-v1.5": ["BAAI/bge-base-en-v1.5", "ST"],
        "GloVe 6B 300d": ["sentence-transformers/average_word_embeddings_glove.6B.300d", "ST"],
        "potion-base-8M": ["minishlab/potion-base-8M", "M2V"],
    }

    # Load the dataset
    ds = load_dataset("wikimedia/wikipedia", data_files="20231101.en/train-00000-of-00041.parquet")["train"]
    texts = ds["text"][:n_texts]

    summarized_results = [
        {"Model": "potion-base-2M", "Average score": 44.77, "Samples per second": None, "Params (Million)": 1.875},
        {"Model": "GloVe 6B 300d", "Average score": 42.36, "Samples per second": None, "Params (Million)": 120.000},
        {"Model": "potion-base-4M", "Average score": 48.23, "Samples per second": None, "Params (Million)": 3.750},
        {"Model": "all-MiniLM-L6-v2", "Average score": 56.09, "Samples per second": None, "Params (Million)": 23.000},
        {"Model": "potion-base-8M", "Average score": 50.03, "Samples per second": None, "Params (Million)": 7.500},
        {"Model": "bge-base-en-v1.5", "Average score": 63.56, "Samples per second": None, "Params (Million)": 109.000},
        {"Model": "M2V_base_output", "Average score": 45.34, "Samples per second": None, "Params (Million)": 7.500},
        {"Model": "BPEmb-50k-300d", "Average score": 37.78, "Samples per second": None, "Params (Million)": 15.000},
        {"Model": "potion-base-32M", "Average score": 51.66, "Samples per second": None, "Params (Million)": 32.300},
    ]

    timings = {}

    for name, info in models.items():
        timing = benchmark_model(name, info, texts)
        timings[name] = timing
        # Update summarized results
        for result in summarized_results:
            if result["Model"] == name:
                result["Samples per second"] = timing["docs_per_second"]

    # Set potion-base-8M as the reference speed for the other potion models
    potion_base_8m_speed = next(
        result["Samples per second"] for result in summarized_results if result["Model"] == "potion-base-8M"
    )
    for model_name in ["M2V_base_output", "potion-base-2M", "potion-base-4M", "potion-base-32M"]:
        for result in summarized_results:
            if result["Model"] == model_name:
                result["Samples per second"] = potion_base_8m_speed

    # Ensure save_path is a directory
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save timings to JSON
    json_path = save_dir / "speed_benchmark_results.json"
    with open(json_path, "w") as file:
        json.dump(timings, file, indent=4)

    # Create and save the plot
    df = pd.DataFrame(summarized_results)
    plot = make_plot(df)
    plot_path = save_dir / "speed_vs_mteb_plot.png"
    plot.save(plot_path, width=12, height=10)

    logger.info(f"Timings saved to {json_path}")
    logger.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark text embedding models and generate a plot.")
    parser.add_argument(
        "--save-path", type=str, required=True, help="Directory to save the benchmark results and plot."
    )
    parser.add_argument(
        "--n-texts", type=int, default=100_000, help="Number of texts to use from the dataset for benchmarking."
    )
    args = parser.parse_args()

    main(save_path=args.save_path, n_texts=args.n_texts)
