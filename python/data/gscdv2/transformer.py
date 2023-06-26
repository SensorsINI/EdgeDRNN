__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import os
import torch
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class StandardScaler(object):
    def __init__(self):
        self.mean = 0
        self.std = 1
    
    def fit(self, dataloader: DataLoader):
        """
        x (Tensor): Tensor of shape (*, Feature)
        """
        # Calculate mean
        for batch_idx, (data, _) in enumerate(dataloader):
            n_features = data.size[-1]
            temp = data.view(-1, n_features)
            if batch_idx == 0:
                num = temp.size(0)
                sum = temp
            else:
                temp = data.view(-1, n_features)
                num += temp.size(0)
                sum += temp
        self.mean = sum / num