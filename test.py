# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from tqdm import tqdm

from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR


def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use config.py for machine-dependent paths, e.g. DATA_DIR and MODEL_SAVE_DIR.
    test_data = CIFAR10Dataset(DATA_DIR, Subset.TEST, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepClassifier(resnet18(weights=None, num_classes=test_data.num_classes()))
    model.load(args.path_to_trained_model)
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(test_data_loader, desc="Test"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            test_metric.update(outputs, targets)

    avg_test_loss = test_loss / max(1, len(test_data_loader))

    print(f"Test loss: {avg_test_loss:.4f}")
    print(test_metric)


if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )
    parser.add_argument(
        "-p",
        "--path_to_trained_model",
        default=str(MODEL_SAVE_DIR / "model_best.pt"),
        type=str,
        help="path to the saved model checkpoint",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.path_to_trained_model = str(MODEL_SAVE_DIR / "model_best.pt")

    test(args)
