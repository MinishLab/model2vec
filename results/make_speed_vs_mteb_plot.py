"""Script to benchmark the speed of various text embedding models and generate a plot of the MTEB score vs samples per second."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from sentence_transformers import SentenceTransformer

from model2vec import StaticModel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


MAX_LENGTH = 512


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """Encode a list of sentences."""
        ...


@dataclass(frozen=True)
class Result:
    name: str
    path: str
    kind: str
    family: str
    score: float
    params_million: float


RESULTS: list[Result] = [
    Result("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "ST", "Transformer", 60.77, 109.0),
    Result("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "ST", "Transformer", 55.93, 23.0),
    Result(
        "GloVe 6B 300d",
        "sentence-transformers/average_word_embeddings_glove.6B.300d",
        "ST",
        "Word embeddings",
        45.82,
        120.0,
    ),
    Result("blink-8M", "minishlab/blink-8m", "M2V", "Model2Vec", 55.35, 7.68),
]

# Cached speeds (samples/sec, best run on CPU over 1k Wikipedia texts, truncated to 512 tokens).
# Re-run with --force-benchmark to update these values.
CACHED_SPEEDS: dict[str, float] = {
    "all-MiniLM-L6-v2": 100.40,
    "bge-base-en-v1.5": 10.72,
    "GloVe 6B 300d": 480.87,
    "blink-8M": 218288.95,
}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e8e7e3"
FAMILY_COLORS: dict[str, str] = {
    "Model2Vec": "#2a78d6",
    "Transformer": "#eb6834",
    "Word embeddings": "#1baf7a",
}


def _marker_area(params_million: float) -> float:
    """Marker area in points², proportional to the number of parameters."""
    return 40 + 6 * params_million


def _format_speed(value: float, _: Any) -> str:
    """Format a tick on the speed axis, e.g. 10k."""
    for divisor, suffix in ((1e6, "M"), (1e3, "k")):
        if value >= divisor:
            return f"{value / divisor:g}{suffix}"
    return f"{value:g}"


def make_plot(speeds: dict[str, float]) -> plt.Figure:
    """Create a plot of the MTEB score vs samples per second."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "path",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for result in sorted(RESULTS, key=lambda result: -result.params_million):
        speed = speeds[result.name]
        area = _marker_area(result.params_million)
        ax.scatter(
            speed,
            result.score,
            s=area,
            color=FAMILY_COLORS[result.family],
            edgecolors=SURFACE,
            linewidths=2,
            zorder=3,
        )
        ax.annotate(
            result.name,
            (speed, result.score),
            xytext=(np.sqrt(area) / 2 + 6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(5, 2.5e6)
    ax.xaxis.set_major_locator(FixedLocator([10**exponent for exponent in range(1, 7)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_format_speed))
    ax.set_ylim(44, 63)
    ax.set_yticks(range(45, 65, 5))

    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0, pad=6)

    ax.set_xlabel("Samples per second (log scale)", color=TEXT_SECONDARY, fontsize=10, labelpad=8)
    ax.set_ylabel("Average MTEB score", color=TEXT_SECONDARY, fontsize=10, labelpad=8)

    fig.text(0.06, 0.95, "MTEB score vs. encoding speed", fontsize=14, weight="bold", color=TEXT_PRIMARY)
    fig.text(
        0.06,
        0.905,
        f"CPU, 1k Wikipedia articles truncated to {MAX_LENGTH} tokens. Circle area is proportional to the number of parameters.",
        fontsize=9.5,
        color=TEXT_SECONDARY,
    )

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=color, markeredgecolor=SURFACE, label=family)
        for family, color in FAMILY_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handletextpad=0.3,
        columnspacing=1.6,
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.11)
    return fig


def load_model(path: str, kind: str) -> Encoder:
    """Load a model of the given kind."""
    if kind == "ST":
        return SentenceTransformer(path, device="cpu")
    return StaticModel.from_pretrained(path)


def benchmark_model(name: str, path: str, kind: str, texts: list[str], n_runs: int, min_seconds: float) -> float:
    """Benchmark a single model, returning the best samples per second.

    Runs at least `n_runs` times, and keeps running until `min_seconds` have passed in total.
    Static models are truncated to `MAX_LENGTH` tokens, like the transformer models.
    """
    logger.info("Starting %s", name)
    model = load_model(path, kind)
    encode_kwargs = {"max_length": MAX_LENGTH} if kind == "M2V" else {}
    model.encode(texts[:16], **encode_kwargs)

    best_seconds = float("inf")
    total_seconds = 0.0
    runs = 0
    while runs < n_runs or total_seconds < min_seconds:
        start = perf_counter()
        model.encode(texts, **encode_kwargs)
        elapsed = perf_counter() - start
        best_seconds = min(best_seconds, elapsed)
        total_seconds += elapsed
        runs += 1

    docs_per_second = len(texts) / best_seconds
    logger.info("%s: %.2f docs per second (best of %d runs: %.4fs)", name, docs_per_second, runs, best_seconds)
    return docs_per_second


def main(save_path: str, n_texts: int, n_runs: int, min_seconds: float, force_benchmark: bool) -> None:
    """Benchmark text embedding models and generate a plot."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    speeds = dict(CACHED_SPEEDS)
    if force_benchmark:
        ds = load_dataset("wikimedia/wikipedia", data_files="20231101.en/train-00000-of-00041.parquet")["train"]
        texts = ds["text"][:n_texts]
        for result in RESULTS:
            speeds[result.name] = benchmark_model(result.name, result.path, result.kind, texts, n_runs, min_seconds)
        logger.info("Updated speeds: %s", speeds)

    fig = make_plot(speeds)
    plot_path = save_dir / "speed_vs_mteb_plot.svg"
    fig.savefig(plot_path, facecolor=SURFACE)
    logger.info("Plot saved to %s", plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark text embedding models and generate a plot.")
    parser.add_argument(
        "--save-path", type=str, required=True, help="Directory to save the benchmark results and plot."
    )
    parser.add_argument(
        "--n-texts", type=int, default=1_000, help="Number of texts to use from the dataset for benchmarking."
    )
    parser.add_argument(
        "--n-runs", type=int, default=3, help="Minimum number of timed runs per model; the best is kept."
    )
    parser.add_argument(
        "--min-seconds", type=float, default=2.0, help="Keep timing a model until this many seconds have passed."
    )
    parser.add_argument(
        "--force-benchmark",
        action="store_true",
        help="Re-run the speed benchmark even if cached results exist.",
    )
    args = parser.parse_args()

    main(
        save_path=args.save_path,
        n_texts=args.n_texts,
        n_runs=args.n_runs,
        min_seconds=args.min_seconds,
        force_benchmark=args.force_benchmark,
    )
