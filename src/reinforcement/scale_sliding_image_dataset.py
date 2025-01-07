import os
from pathlib import Path
from typing import Iterator
from attr import dataclass
import torch
from torchvision import transforms
from PIL import Image


@dataclass
class SlidingImageMetadata:
    scale: int
    x: int
    y: int
    idx: int


class ScaleSlidingImagesDataset(torch.utils.data.IterableDataset):
    files: list[Path]
    """Path to the dataset."""
    transform: transforms.Compose
    """Transform to apply to the images."""
    window_size: int
    """Size of the sliding window."""
    stride: int
    """Stride of the sliding window."""
    scale_step: int
    """Resize step for each scale."""

    def __init__(
        self,
        files: list[Path],
        transform: transforms.Compose,
        *,
        window_size: int,
        stride: int,
        scale_step: int = 1,
    ):
        self.files = files
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.scale_step = scale_step

    def __len__(self):
        return len(self.files)

    def __iter__(self) -> Iterator:
        for idx in range(len(self)):
            yield from self[idx]

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])

        for scale, resized_image in self._resize_img(image):
            for x, y, window in self._window_slide(resized_image):
                yield (
                    self.transform(window),
                    SlidingImageMetadata(scale=scale, x=x, y=y, idx=idx).__dict__,
                )

    def _resize_img(self, img: Image.Image) -> Iterator[tuple[int, Image.Image]]:
        """Resize the image to different scales.

        Args:
            img (Image.Image): Image to resize.

        Yields:
            Iterator[tuple[float, Image.Image]]: Iterator of the scale and the resized image.
        """
        size = min(img.width // 2, img.height // 2)

        for scale in range(size // self.window_size, 0, -self.scale_step):
            yield scale, img.resize((img.width // scale, img.height // scale))

    def _window_slide(self, img: Image.Image) -> Iterator[tuple[int, int, Image.Image]]:
        """Slide a window over the image.

        Args:
            img (Image.Image): Image to slide the window over.

        Yields:
            Iterator[tuple[int, int, Image.Image]]: Iterator of the x, y coordinates and the windowed image.
        """

        for y in range(0, img.height - self.window_size + 1, self.stride):
            for x in range(0, img.width - self.window_size + 1, self.stride):
                yield x, y, img.crop((x, y, x + self.window_size, y + self.window_size))


class ScaleSlidingImagesDirDataset(ScaleSlidingImagesDataset):
    def __init__(
        self,
        dir_path: Path,
        transform: transforms.Compose,
        *,
        window_size: int,
        stride: int,
        scale_step: int = 1,
    ):
        super().__init__(
            self._get_image_files(dir_path),
            transform,
            window_size=window_size,
            stride=stride,
            scale_step=scale_step,
        )

    def _get_image_files(self, path: Path) -> list[Path]:
        return [
            path / f
            for f in os.listdir(path)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".gif", ".tiff")
            )
        ]