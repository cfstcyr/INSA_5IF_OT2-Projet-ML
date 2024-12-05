from datetime import datetime
from typing import Any

import torch
from tqdm import tqdm
from src.config import config
from src.data import CLASSES, get_dataloader, get_datasets
from src.device import get_device
from src.net import Net
import os

from src.results import Result, TestResults


def test(
    net: Net, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> TestResults:
    result: Result = Result(correct=0, total=0)
    labels_results: dict[Any, Result] = {
        label: Result(correct=0, total=0) for label in CLASSES
    }

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Test"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            result.total += labels.size(0)
            result.correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                labels_results[CLASSES[label]].total += 1
                labels_results[CLASSES[label]].correct += int(label == prediction)

    return TestResults(result=result, labels_results=labels_results)


if __name__ == "__main__":
    results_name = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_name = sorted(os.listdir(config.models_dir))[-1]
    model_dir = config.models_dir / model_name
    results_dir = config.results_dir / model_name

    device = get_device()

    net = Net()
    net.load_state_dict(torch.load(model_dir / "model.pth", weights_only=False))
    net.to(device)

    _, _, test_loader = get_dataloader(*get_datasets())

    test_results = test(net, test_loader, device)

    print(f"Accuracy: {test_results.result.correct / test_results.result.total:.2%}")
    for label, results in test_results.labels_results.items():
        print(f"  {label}: {results.correct / results.total:.2%}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{results_name}.json", "w") as f:
        f.write(test_results.model_dump_json(indent=2))
