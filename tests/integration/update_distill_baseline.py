"""Regenerate the golden baselines used by `test_distill_regression.py`.

Run this deliberately (`make test-integration-update`) after a change that intentionally alters
distillation output (e.g. a new default, a bugfix that shifts embeddings). Then inspect the JSON
diff (`git diff tests/integration/data/`) to confirm the change is expected before committing it --
an unreviewed update here would hide a real regression.

By default this regenerates the baseline for every model in `BASE_MODELS`. Pass one or more short
names (the keys of `BASE_MODELS`, e.g. `minilm`) to only regenerate those, e.g.:

    uv run python -m tests.integration.update_distill_baseline minilm
"""

from __future__ import annotations

import json
import logging
import sys

from tests.integration._distill_metrics import (
    BASE_MODELS,
    baseline_path_for,
    compute_metrics,
    distill_all,
    load_base_model_and_tokenizer,
)

logger = logging.getLogger(__name__)


def update_baseline(model_name: str) -> None:
    """Distill every configured variant of one base model and write their metrics to its baseline file.

    :param model_name: A short name, i.e. a key of `BASE_MODELS`.
    """
    hub_model_name = BASE_MODELS[model_name]
    model, tokenizer = load_base_model_and_tokenizer(hub_model_name)
    distilled = distill_all(model, tokenizer)

    baseline = {
        "base_model": hub_model_name,
        "configs": {name: compute_metrics(static_model) for name, static_model in distilled.items()},
    }

    path = baseline_path_for(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote baseline for '{model_name}' ({hub_model_name}) to {path}")


def main() -> None:
    """Regenerate the baselines requested on the command line, or all of them if none were given."""
    requested = sys.argv[1:] or sorted(BASE_MODELS)
    unknown = [name for name in requested if name not in BASE_MODELS]
    if unknown:
        raise SystemExit(f"Unknown model name(s) {unknown}. Choose from {sorted(BASE_MODELS)}.")

    for model_name in requested:
        update_baseline(model_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
