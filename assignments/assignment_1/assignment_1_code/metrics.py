from abc import ABCMeta, abstractmethod
import torch
from typing import Dict


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0
        self.per_class_accuracies = ...  # dict mapping class name to accuracy

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        # validate input
        if len(prediction.shape) != 2:
            raise ValueError(f"prediction must have shape (batchsize,n_classes), but has shape {prediction.shape}")
        if len(target.shape) != 1:
            raise ValueError(f"target must have shape (batchsize,), but has shape {target.shape}")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"prediction and target must have the same batchsize, but have batchsize {prediction.shape[0]} and {target.shape[0]}")
        if prediction.shape[1] != len(self.classes):
            raise ValueError(f"prediction must have n_classes={len(self.classes)} columns, but has {prediction.shape[1]} columns")
        if not torch.all((target >= 0) & (target < len(self.classes))):
            raise ValueError(f"target values must be between 0 and {len(self.classes)-1}, but target has values between {torch.min(target)} and {torch.max(target)}")
        
        #update measure


        pass

    def __str__(self):
        """
        Return a string representation of the performance including:
        - overall accuracy
        - mean per-class accuracy
        - individual per-class accuracies for all classes
        """

        # TODO implement
        pass

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        # compute accuracy
        if self.n_total == 0:
            return 0.0
        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the mean per-class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        Saves the individual per-class accuracies in self.per_class_accuracies as a dict mapping class name to accuracy.
        """
        if self.n_total == 0:
            return 0.0
        
        valid_classes = 0
        for classname in self.classes:
            if self.total_pred[classname] == 0:
                self.per_class_accuracies[classname] = 0.0
            else:
                self.per_class_accuracies[classname] = self.correct_pred[classname] / self.total_pred[classname]
                valid_classes += 1
        
        if valid_classes == 0:
            return 0.0
        
        return sum(self.per_class_accuracies.values()) / valid_classes


