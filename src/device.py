import logging
import torch


logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    device = torch.device(
        "mps"
        if torch.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(f"Using device: {device}")

    return device
