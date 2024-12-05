from pydantic import BaseModel


class Metadata(BaseModel):
    name: str
    n_epochs: int
    train_loss: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
