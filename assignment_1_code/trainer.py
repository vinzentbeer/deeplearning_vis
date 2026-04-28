import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency

        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        
        self.val_data_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        self.wandb_logger = WandBLogger()

        pass

    def _train_epoch(self, epoch_idx: int, debug:bool=False) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        # Hint: you can use tqdm to have a progress bar for the training loop, e.g.:
        # for batch in tqdm(self.train_data_loader, desc=f"Epoch {epoch_idx}"):

        

        self.model.train()
        train_loss = 0.0

        # https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
        for inputs, targets in self.train_data_loader: 
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            
            if(debug):
                print("loss = " + loss.item())
            
            self.train_metric.update(outputs, targets)

        print(self.train_metric)
        return train_loss, self.train_metric.accuracy(), self.train_metric.per_class_accuracy()

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        
        self.model.eval()
        val_loss = 0.0

        # https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
        with torch.no_grad():
            for inputs, targets in self.val_data_loader: 
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                self.optimizer.step()
                val_loss += loss.item()
                
                self.val_metric.update(outputs, targets)

        print(self.val_metric)
        return val_loss, self.val_metric.accuracy(), self.val_metric.per_class_accuracy()        
        pass

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ## TODO implement

        for epoch in range(self.num_epochs):
            train_loss, train_acc, train_pcacc = self._train_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_loss}, Train Acc: {train_acc}, Train PCAcc: {train_pcacc}")

            if epoch % self.val_frequency == 0:
                val_loss, val_acc, val_pcacc = self._val_epoch(epoch)
                print(f"Epoch {epoch} - Val Loss: {val_loss}, Val Acc: {val_acc}, Val PCAcc: {val_pcacc}")

                # Save the model if mean per class accuracy on validation data set is higher than currently saved best mean per class accuracy.
                # You can use torch.save() to save the model and torch.load() to load it.
                # Make sure to save the model in the training_save_dir folder and give it a meaningful name, e.g. "best_model.pth".
        pass


