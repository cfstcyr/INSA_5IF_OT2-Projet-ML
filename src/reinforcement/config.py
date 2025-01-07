from pathlib import Path
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings


class ReinforcementConfig(BaseSettings):
    base_model_path: FilePath = (
        Path("models") / "model_2025-01-07_08-20-25" / "model.pth"
    )
    # output_model_path: Path = (
    #     Path("models") / "model_2024-12-05_11-33-59" / "reinforced_model.pth"
    # )

    source_images_path: DirectoryPath = Path("data") / "false_positives_sources_v2"
    target_images_path: Path = Path("data") / "false_positives_targets_v2"

    window_size: int = 32
    stride: int = 8

    n_epochs: int = 10


config = ReinforcementConfig()

__all__ = ["config"]
