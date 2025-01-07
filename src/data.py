from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
import torchvision.datasets
from torch.utils.data.sampler import SubsetRandomSampler

from src.config import config

CLASSES = ("noface", "face")

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize(mean=(0,), std=(1,)),
    ]
)

def get_dataset(data_dir: Path) -> torchvision.datasets.VisionDataset:
    return torchvision.datasets.ImageFolder(data_dir, transform=transform)


def get_default_dataloader(dataset: torchvision.datasets.VisionDataset, *, batch_size: int, **kwargs) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(42),
        **kwargs,
    )


def get_datasets(
    train_dir: Path = config.train_dir, test_dir: Path = config.test_dir
) -> tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    return get_dataset(train_dir), get_dataset(test_dir)


def get_dataloaders(
    train_data: torchvision.datasets.VisionDataset,
    test_data: torchvision.datasets.VisionDataset,
    batch_size: int = config.batch_size,
    valid_size: float = config.valid_size,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

    train_sampler = SubsetRandomSampler(
        train_new_idx, generator=torch.Generator().manual_seed(42)
    )
    valid_sampler = SubsetRandomSampler(
        valid_idx, generator=torch.Generator().manual_seed(42)
    )

    train_loader = get_default_dataloader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = get_default_dataloader(train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = get_default_dataloader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
