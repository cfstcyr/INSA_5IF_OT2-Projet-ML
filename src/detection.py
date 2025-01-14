"""Run model detection and compute metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import torch
from tqdm import tqdm
from src.config import config
from src.detection_utils import compute_box_overlap, non_max_suppression, scale_boxes
from src.device import get_device
from src.net import Net
from src.reinforcement.scale_sliding_image_dataset import ScaleSlidingImagesDataset
from src.data import transform


@dataclass
class ImageFace:
    x0: int
    y0: int
    x1: int
    y1: int

@dataclass
class ImageSource:
    image: Path
    faces: list[ImageFace] = field(default_factory=list)


@dataclass
class ImageResult:
    source: ImageSource
    noise_ratio: float
    faces_accuracy: float


def collect_dataset(dir: Path) -> Iterator[ImageSource]:
    label_file = dir / "label.txt"

    current: ImageSource | None = None

    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if current:
                    yield current

                current = ImageSource(image=dir / "images" / line[1:].strip())
            elif current:
                x0, y0, x1, y1 = map(int, line.split())
                current.faces.append(ImageFace(x0, y0, x1, y1))
            else:
                raise ValueError("Unexpected line")
    
    if current:
        yield current


def detection(net: Net, device: torch.device, dataset_dir: Path, *, window_size: int = 32, stride: int = 4, threshold: float = .30) -> Iterator[ImageResult]:
    all_images = list(collect_dataset(dataset_dir))

    for image_source in (progress := tqdm(all_images)):
        progress.set_description(f"Processing {image_source.image.name}")

        dataset = ScaleSlidingImagesDataset(
            [image_source.image],
            transform=transform,
            window_size=window_size,
            stride=stride,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
        )

        found: list[tuple[float, float, float, float]] = []

        with torch.no_grad():
            for batch in dataloader:
                data, meta = batch
                data = data.to(device)

                outputs = net(data)
                _, predicted = torch.max(outputs, 1)
                confidences = torch.nn.functional.softmax(outputs, dim=1)

                for pred, conf, scale, x, y, idx in zip(predicted, confidences, meta["scale"], meta["x"], meta["y"], meta["idx"]):
                    if pred != 1:
                        continue
                        
                    _, conf = conf

                    found.append((scale, x, y, conf))
        
        scaled_boxes = scale_boxes(found, window_size=window_size)

        scaled_boxes_index = non_max_suppression(
            [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax, _ in scaled_boxes],
            [confidence for _, _, _, _, confidence in scaled_boxes],
            1e-99,
        )

        print(f"Found {len(scaled_boxes_index[0])} faces (out of {len(image_source.faces)})")
        noise_count: int = 0
        face_accuracy_sum: float = 0

        for i, box_confidence in zip(scaled_boxes_index[0], scaled_boxes_index[1]):
            xmin, ymin, xmax, ymax, confidence = scaled_boxes[i]

            best_score: float = 0
            for i, face in enumerate(image_source.faces):
                overlap_score = compute_box_overlap(
                    (xmin, ymin, xmax, ymax),
                    (face.x0, face.y0, face.x1, face.y1),
                )

                if overlap_score > best_score:
                    best_score = overlap_score

            if best_score < threshold:
                noise_count += 1
            else:
                face_accuracy_sum += best_score

        noise_ratio = noise_count / len(scaled_boxes_index[0])
        face_accuracy = face_accuracy_sum / (len(scaled_boxes_index[0]) - noise_count)

        yield ImageResult(
            source=image_source,
            noise_ratio=float(noise_ratio),
            faces_accuracy=float(face_accuracy),
        )


if __name__ == "__main__":
    model_path = config.models_dir / "model_2025-01-07_09-45-42" / "model.pth"
    dataset_dir = config.data_dir / "detection_dataset"

    device = get_device()

    net = Net()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.to(device)

    all_results: list[ImageResult] = []

    def agg_results():
        noise_ratio_sum = 0
        faces_accuracy_sum = 0

        for res in all_results:
            noise_ratio_sum += res.noise_ratio
            faces_accuracy_sum += res.faces_accuracy

        print(f"Average noise ratio: {noise_ratio_sum / len(all_results)}")
        print(f"Average faces accuracy: {faces_accuracy_sum / len(all_results)}")

    for i, res in enumerate(detection(net, device, dataset_dir)):
        all_results.append(res)
        print(1 - res.noise_ratio, res.faces_accuracy)

        if i > 10:
            break
    
    agg_results()