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


def get_datasets(
    train_dir: Path = config.train_dir, test_dir: Path = config.test_dir
) -> tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    return train_data, test_data


def get_dataloader(
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

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        generator=torch.Generator().manual_seed(42),
    )

    return train_loader, valid_loader, test_loader
