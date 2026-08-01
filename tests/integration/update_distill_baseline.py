from __future__ import annotations

import json
import logging
import sys

from tests.integration.distill_metrics import (
    BASE_MODELS,
    baseline_path_for,
    compute_metrics,
    distill_all,
    load_base_model_and_tokenizer,
)

logger = logging.getLogger(__name__)


def update_baseline(model_name: str) -> None:
    """Distill every configured variant of one base model and write their metrics to its baseline file.

    :param model_name: A model identifier on Hugging Face.
    """
    model, tokenizer = load_base_model_and_tokenizer(model_name)
    distilled = distill_all(model, tokenizer)

    baseline = {
        "base_model": model_name,
        "configs": {name: compute_metrics(static_model) for name, static_model in distilled.items()},
    }

    path = baseline_path_for(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote baseline for '{model_name}' to {path}")


def main() -> None:
    """Regenerate the baselines requested on the command line, or all of them if none were given."""
    requested = sys.argv[1:] or sorted(BASE_MODELS)

    for model_name in requested:
        update_baseline(model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
