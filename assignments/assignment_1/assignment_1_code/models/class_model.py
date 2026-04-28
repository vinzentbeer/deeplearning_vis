import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''

        filename = "model.pt" if suffix is None else f"model_{suffix}.pt"
        save_path = save_dir / filename
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        self.load_state_dict(torch.load(path), weights_only=True)
        self.eval()
        
        pass