from pathlib import Path
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    data_dir: DirectoryPath = Path("data")
    train_dir: DirectoryPath = data_dir / "train_images"
    test_dir: DirectoryPath = data_dir / "test_images"

    models_dir: DirectoryPath = Path("models")
    results_dir: DirectoryPath = Path("results")

    batch_size: int = 32
    valid_size: float = 0.2

    n_epochs: int = 20

    seed: int = 42


config = Config()

__all__ = ["config"]
