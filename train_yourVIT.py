# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from assignment_1_code.models.class_model import (
    DeepClassifier,
)  # etc. change to your model
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.cnn import YourCNN
from assignment_1_code.models.vit import VisionTransformer
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR

from torchvision.models import resnet18
from datetime import datetime

def create_run_name(model_name, optimizer_name, lr, scheduler_name, num_epochs, batch_size=None):
    """
    Create a descriptive run name from hyperparameters.
    Example: CNN_adamw_lr1e-03_exponential_20ep_bs128
    """
    name = f"{model_name}_{optimizer_name}_lr{lr:.0e}_{scheduler_name}_{num_epochs}ep"
    if batch_size:
        name += f"_bs{batch_size}"
    return name

def train(args):

    # Implement this function so that it trains a specific model as described in the instruction.md file
    # feel free to change the code snippets given here, they are just to give you an initial structure
    # but do not have to be used if you want to do it differently
    # For device handling you can take a look at pytorch documentation 

    
    
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(size=32, padding=4),
            #v2.RandomAutocontrast(),
            #v2.RandomAdjustSharpness(sharpness_factor=2.0),
            #v2.RandomRotation(degrees=15),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    #print debug path
    print(f"Using data directory: {DATA_DIR}")
    #print resolved path
    print(f"Resolved data directory: {DATA_DIR.resolve()}")

    # Use config.py for all machine-dependent paths, e.g. DATA_DIR.
    train_data = CIFAR10Dataset(DATA_DIR, Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(DATA_DIR, Subset.VALIDATION, transform=val_transform)
    #test_data = CIFAR10Dataset(DATA_DIR, Subset.TEST, transform=val_transform)

    # print first 10 training label classes as dbg 
    print("First 10 training label classes:", train_data.labels[:10])

    print(train_data.num_classes())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print selected device
    print(f"Using device: {device}")

    model = DeepClassifier(
        VisionTransformer(
            embed_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            patch_size=4,
            num_patches=64,
            num_channels=3,
            num_classes=train_data.num_classes(),
            dropout=0.2,
        )
    )
    lr = 3e-4 #according to tutorial for the vit

    model.to(device)
    
    # Optimizer configuration
    optimizer_name = "adamw"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True) 
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5
    batch_size = 128

    model_save_dir = Path(MODEL_SAVE_DIR)
    model_save_dir.mkdir(exist_ok=True)

    # Learning rate scheduler configuration
    scheduler_name = "multistep=0.1"
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    model_name = "ViT"
   
    # Create W&B run name from hyperparameters
    run_name = create_run_name(
        model_name=model_name,
        optimizer_name=optimizer_name,
        lr=lr,
        scheduler_name=scheduler_name,
        num_epochs=args.num_epochs,
        batch_size=batch_size
    )

    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        model_save_dir,
        batch_size=batch_size,
        val_frequency=val_frequency,
        run_name=run_name,
    )
    trainer.train()


if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 40

    train(args)
