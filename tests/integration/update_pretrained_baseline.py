from __future__ import annotations

import json
import logging
import sys

from tests.integration.pretrained_model_metrics import (
    PRETRAINED_MODELS,
    baseline_path_for,
    compute_metrics,
    load_static_model,
)

logger = logging.getLogger(__name__)


def update_baseline(model_name: str) -> None:
    """Load one published model and write its metrics snapshot to its baseline file.

    :param model_name: A model identifier on Hugging Face.
    """
    model = load_static_model(model_name)
    baseline = {"model": model_name, "metrics": compute_metrics(model)}

    path = baseline_path_for(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote baseline for '{model_name}' to {path}")


def main() -> None:
    """Regenerate the baselines requested on the command line, or all of them if none were given."""
    requested = sys.argv[1:] or list(PRETRAINED_MODELS)

    for model_name in requested:
        update_baseline(model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
