
import os
from pathlib import Path
import tempfile

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_API_KEY", "dummy")

import torch
import wandb

from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.trainer import ImgClassificationTrainer


wandb.login = lambda *args, **kwargs: None


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 16, num_classes: int = 10) -> None:
        self.data = torch.randn(size, 3, 32, 32)
        self.targets = torch.randint(0, num_classes, (size,))
        self.classes = [f"class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


def main() -> None:
    train_data = DummyDataset()
    val_data = DummyDataset()

    model = DeepClassifier(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    save_dir = Path(tempfile.mkdtemp())
    trainer = ImgClassificationTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        train_metric=Accuracy(train_data.classes),
        val_metric=Accuracy(val_data.classes),
        train_data=train_data,
        val_data=val_data,
        device=torch.device("cpu"),
        num_epochs=1,
        training_save_dir=save_dir,
        batch_size=4,
        val_frequency=1,
    )

    trainer.train()

    checkpoint = save_dir / "model_best.pt"
    assert checkpoint.exists(), f"missing checkpoint: {checkpoint}"

    reloaded = DeepClassifier(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10)))
    reloaded.load(checkpoint)

    print("SMOKE_TEST_OK")
    print(checkpoint)


if __name__ == "__main__":
    main()