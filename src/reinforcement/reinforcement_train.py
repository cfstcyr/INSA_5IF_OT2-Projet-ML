import torch
from tqdm import tqdm
from src.data import get_dataloaders, get_dataset, get_datasets, get_default_dataloader
from src.device import get_device
from src.metadata import TrainMetadata
from src.net import Net
from torch import nn, optim

from src.reinforcement.config import config
from src.train import train


# def validate(
#     net: Net,
#     data_loader: torch.utils.data.DataLoader,
#     criterion: nn.Module,
#     device: torch.device,
# ) -> tuple[float, float]:
#     net.eval()

#     val_loss: float = 0.0
#     correct: int = 0
#     total: int = 0

#     with torch.no_grad():
#         for data in tqdm(data_loader, desc="Validation", leave=False):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = net(inputs)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)

#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     val_loss /= len(data_loader)
#     accuracy = correct / total

#     return val_loss, accuracy


# def reinforcement_train(base_model: Net, train_loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[TrainMetadata, dict]:
#     net = base_model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#     train_losses = []

#     for _ in tqdm(range(config.n_epochs), desc="Epochs"):
#         running_loss = 0.0

#         for data in (batch_progress := tqdm(train_loader, desc="Batches")):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()

#             outputs = net(inputs)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             batch_progress.set_postfix(loss=f"{running_loss / len(train_loader):.3f}")
        
#         train_losses.append(running_loss / len(train_loader))

#     return TrainMetadata(
#         name="reinforcement",
#         n_epochs=config.n_epochs,
#         train_loss=train_losses,
#         val_loss=[],
#         val_accuracy=[],
#     ), net.state_dict()

if __name__ == "__main__":
    source_model = Net()
    source_model.load_state_dict(torch.load(config.base_model_path, weights_only=False))

    print(f"Loaded model from {config.base_model_path}")

    device = get_device()

    _, valid_loader, _ = get_dataloaders(*get_datasets())

    source_dataset = get_dataset(config.target_images_path)
    source_dataloader = get_default_dataloader(source_dataset, batch_size=32)

    metadata, state_dict = train("reinforcement", device, source_dataloader, valid_loader, n_epochs=config.n_epochs)

    torch.save(state_dict, config.output_model_path)

    print(f"Finished training reinforcement model: {metadata}")
    print(f"Saved model to {config.output_model_path}")