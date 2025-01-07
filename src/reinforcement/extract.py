"""
# Extract

This module contains the functions to extract the input data for the reinforcement learning from images.
"""

import os
from typing import Iterator
import torch
from tqdm import tqdm
from src.device import get_device
from src.net import Net
from src.reinforcement.config import config
from PIL import Image
from torchvision import transforms
import shutil

from src.reinforcement.scale_sliding_image_dataset import (
    ScaleSlidingImagesDirDataset,
    SlidingImageMetadata,
)


def _extract_faces(
    dataloader: torch.utils.data.DataLoader, net: Net, device: torch.device
) -> Iterator[SlidingImageMetadata]:
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting faces", total=0):
            data, meta = data
            data = data.to(device)

            outputs = net(data)
            _, predicted = torch.max(outputs, 1)

            for prediction, scale, x, y, idx in zip(
                predicted, meta["scale"], meta["x"], meta["y"], meta["idx"]
            ):
                if prediction != 1:
                    continue

                yield SlidingImageMetadata(
                    scale=scale.item(),
                    x=x.item(),
                    y=y.item(),
                    idx=idx.item(),
                )


def extract_input():
    device = get_device()

    net = Net()
    net.load_state_dict(torch.load(config.base_model_path, weights_only=False))
    net.to(device)

    source_dataset = ScaleSlidingImagesDirDataset(
        config.source_images_path,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
        window_size=config.window_size,
        stride=config.stride,
        scale_step=1,
    )
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=32,
        shuffle=False,
    )

    target_images_path = config.target_images_path / "0"
    target_images_path.mkdir(parents=True, exist_ok=True)

    for false_positive in _extract_faces(source_dataloader, net, device):
        image_path = source_dataset.files[false_positive.idx]
        image = Image.open(image_path)
        image = image.resize(
            (
                image.width // false_positive.scale,
                image.height // false_positive.scale,
            )
        )
        image = image.crop(
            (
                false_positive.x,
                false_positive.y,
                false_positive.x + config.window_size,
                false_positive.y + config.window_size,
            )
        )

        image.save(
            target_images_path
            / f"{image_path.stem}_{false_positive.scale}_{false_positive.x},{false_positive.y}.png"
        )


if __name__ == "__main__":
    extract_input()

    source_dir = "data/false_positives_targets_v2/0"
    target_dir = "data/train_images_all/0"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        shutil.copy(source_file, target_file)
