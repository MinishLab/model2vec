from __future__ import annotations

from logging import getLogger

import torch
from packaging import version

logger = getLogger(__name__)


def select_optimal_device(device: str | None) -> str:
    """
    Get the optimal device to use based on backend availability.

    For Torch versions >= 2.8.0, MPS is disabled due to known performance regressions.

    :param device: The device to use. If this is None, the device is automatically selected.
    :return: The selected device.
    :raises RuntimeError: If MPS is requested on a PyTorch version where it is disabled.
    """
    # Get the torch version and check if MPS is broken
    torch_version = version.parse(torch.__version__.split("+")[0])
    mps_broken = torch_version >= version.parse("2.8.0")

    if device:
        if device == "mps" and mps_broken:
            raise RuntimeError(
                f"MPS is disabled for PyTorch {torch.__version__} due to known performance regressions. "
                "Please use CPU or CUDA instead."
            )
        else:
            return device

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        if mps_broken:
            logger.warning(
                f"MPS is available but PyTorch {torch.__version__} has known performance regressions. "
                "Falling back to CPU."
            )
            device = "cpu"
        else:
            device = "mps"
    else:
        device = "cpu"

    logger.info(f"Automatically selected device: {device}")
    return device
