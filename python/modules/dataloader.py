__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

from torch.utils import data
from modules.dataset import AmproDataset
from project import Project


class DataLoader:
    def __init__(self, proj: Project):
        # Get Arguments
        batch_size = proj.batch_size
        batch_size_test = proj.batch_size_test

        # Split Dataset
        csv_paths_train = ['./data/rachel_pd1.csv', './data/rachel_pd2.csv', './data/rachel_pd3.csv']
        csv_paths_dev = ['./data/rachel_pd4.csv']
        csv_paths_test = ['./data/rachel_pd5.csv']

        # Create PyTorch Dataset
        train_set = AmproDataset(proj, csv_paths_train, 'train')
        dev_set = AmproDataset(proj, csv_paths_dev, 'dev', train_set.mean, train_set.std)
        test_set = AmproDataset(proj, csv_paths_test, 'test', train_set.mean, train_set.std)

        # Create PyTorch dataloaders for train and dev set
        num_workers = int(proj.num_cpu_threads / 4)
        self.train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers)
        self.dev_loader = data.DataLoader(dataset=dev_set, batch_size=batch_size_test, shuffle=False,
                                          num_workers=num_workers)
        self.test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False,
                                           num_workers=num_workers)
