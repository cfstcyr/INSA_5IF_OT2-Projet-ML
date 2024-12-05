from datetime import datetime
import logging
import torch
from torch import nn, optim
from tqdm import tqdm
from src.config import config
from src.data import get_dataloader, get_datasets
from src.device import get_device
from src.metadata import Metadata
from src.net import Net


logger = logging.getLogger(__name__)


def validate(
    net: Net,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    net.eval()

    val_loss: float = 0.0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Validation", leave=False):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(data_loader)
    accuracy = correct / total

    return val_loss, accuracy


def train(
    name: str,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
) -> tuple[Metadata, dict]:
    net = Net().to(device)

    logger.info(f"Training {name}...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")

    for _ in tqdm(range(config.n_epochs), desc="Epochs"):
        running_loss = 0.0

        for data in (batch_progress := tqdm(train_loader, desc="Batches")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_progress.set_postfix(loss=f"{running_loss / len(train_loader):.3f}")

        val_loss, val_accuracy = validate(net, valid_loader, criterion, device)

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    metadata = Metadata(
        name=name,
        n_epochs=config.n_epochs,
        train_loss=train_losses,
        val_loss=val_losses,
        val_accuracy=val_accuracies,
    )

    logger.info(f"Finished training {name}: {metadata}")

    return metadata, net.state_dict()


if __name__ == "__main__":
    name = f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_dir = config.models_dir / name

    device = get_device()

    train_loader, valid_loader, _ = get_dataloader(*get_datasets())

    metadata, state_dict = train(
        name, device, train_loader, valid_loader
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "metadata.json", "w") as f:
        f.write(metadata.model_dump_json(indent=2))

    torch.save(state_dict, model_dir / "model.pth")

    logger.info(f"Saved model {name} to {model_dir}")
