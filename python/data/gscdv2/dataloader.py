__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import torch
from torch.utils import data
from project import Project


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


class MyCollator(object):
    def __init__(self, train_set):
        self.train_set = train_set
        # Get Labels
        self.labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
        del self.train_set

    def __call__(self, batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]


class DataLoader:
    def __init__(self, proj: Project):
        # Get Arguments
        batch_size = proj.batch_size
        batch_size_test = proj.batch_size_test

        # Create Collector
        collate_fn = MyCollator(proj.train_set)

        # Create PyTorch dataloaders for train and dev set
        if proj.accelerator == "gpu":
            # num_workers = int(proj.num_cpu_threads / 4)
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        self.train_loader = data.DataLoader(
            proj.train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.dev_loader = data.DataLoader(
            proj.dev_set,
            batch_size=batch_size_test,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.test_loader = data.DataLoader(
            proj.test_set,
            batch_size=batch_size_test,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

# waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
