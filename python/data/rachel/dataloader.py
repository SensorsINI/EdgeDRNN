__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import importlib

from torch.utils import data
from project import Project


class DataLoader:
    def __init__(self, proj: Project):
        # Get Arguments
        batch_size = proj.batch_size
        batch_size_test = proj.batch_size_test

        # Create PyTorch Dataset
        try:
            mod_dataset = importlib.import_module('data.' + proj.dataset_name + '.dataset')
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please select a supported dataset or check your name spelling.')
        setattr(self, "train_set", mod_dataset.AmproDataset(self, "training"))
        setattr(self, "dev_set", mod_dataset.AmproDataset(self, "validation"))
        setattr(self, "test_set", mod_dataset.AmproDataset(self, "testing"))

        # Create PyTorch dataloaders for train and dev set
        if proj.accelerator == "gpu":
            num_workers = int(proj.num_cpu_threads / 4)
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        # Create PyTorch dataloaders for train and dev set
        self.train_loader = data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=batch_size_test,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.test_loader = data.DataLoader(
            self.test_set,
            batch_size=batch_size_test,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
