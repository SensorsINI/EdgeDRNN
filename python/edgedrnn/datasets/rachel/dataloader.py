__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import importlib

import torch
from torch.utils import data

from edgedrnn.config import ProjectConfig


class DataLoaders:
    def __init__(self, proj: ProjectConfig):
        # Get Arguments
        batch_size = proj.batch_size
        batch_size_test = proj.batch_size_test

        # Create PyTorch Dataset
        try:
            mod_dataset = importlib.import_module(
                f"edgedrnn.datasets.{proj.dataset_name}.dataset"
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please select a supported dataset or check your name spelling."
            )
        self.train_set = mod_dataset.AmproDataset(proj, "training")
        self.dev_set = mod_dataset.AmproDataset(proj, "validation")
        self.test_set = mod_dataset.AmproDataset(proj, "testing")

        # Create PyTorch dataloaders for train and dev set
        num_workers = proj.num_workers
        pin_memory = (
            proj.device in {"auto", "cuda", "gpu"} and torch.cuda.is_available()
        )
        generator = torch.Generator().manual_seed(proj.seed)
        loader_options = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
        }

        # Create PyTorch dataloaders for train and dev set
        self.train_loader = data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            **loader_options,
        )

        self.dev_loader = data.DataLoader(
            self.dev_set,
            batch_size=batch_size_test,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )

        self.test_loader = data.DataLoader(
            self.test_set,
            batch_size=batch_size_test,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )


DataLoader = DataLoaders
