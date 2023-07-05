__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from project import Project
from typing import List


def collect_cps_data(filepath, ctxt_size, pred_size):
    # Load dataframe
    df = pd.read_csv(filepath, skiprows=28)

    # Collect data
    angleD = df.angleD.to_numpy()
    angle_cos = df.angle_cos.to_numpy()
    angle_sin = df.angle_sin.to_numpy()
    position = df.position.to_numpy()
    positionD = df.positionD.to_numpy()
    target_equilibrium = df.target_equilibrium.to_numpy()
    target_position = df.target_position.to_numpy()

    # position_desired_knee = df.position_desired_knee.to_numpy()
    # velocity_desired_knee = df.velocity_desired_knee.to_numpy()
    # position_error_knee = angle_sin - position_desired_knee
    # velocity_error_knee = positionD - velocity_desired_knee

    # Train Data
    train_data = []
    train_data.append(angleD)
    train_data.append(angle_cos)
    train_data.append(angle_sin)
    train_data.append(position)
    train_data.append(positionD)
    train_data.append(target_equilibrium)
    train_data.append(target_position)
    train_data = np.vstack(train_data).transpose()

    # SPI Data for MiniZed-BBB Debugging
    spi_data = []
    spi_data.append(angleD)
    spi_data.append(angle_cos)
    spi_data.append(angle_sin)
    spi_data.append(position)
    spi_data.append(positionD)
    spi_data.append(target_equilibrium)
    spi_data.append(target_position)
    spi_data.append(position)

    # Collect label
    raw_labels = [df.Q.to_numpy()]
    raw_labels = np.vstack(raw_labels).transpose()

    # Get CPS sample
    cps_test_sample = spi_data
    cps_train_sample = train_data[1:-1, :]

    # Split train data
    train_data_len = cps_train_sample.shape[0]

    features = []
    labels = []
    for i in range(0, train_data_len - pred_size - ctxt_size):
        features.append(cps_train_sample[i:i + ctxt_size, :])
        labels.append(raw_labels[i + pred_size:i + ctxt_size + pred_size])
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)

    return cps_test_sample, features, labels


class CPSDataset(data.Dataset):
    def __init__(self, proj: Project, subset: str):
        """
        param name: 'train', 'dev' or 'test'
        """
        # csv_paths: List, name: str, mean: float=None, std: float=None
        # Dataset Paths
        train_folder = "./data/cps/Train"
        dev_folder = "./data/cps/Validate"
        test_folder = "./data/cps/Test"
        csv_paths_train = get_csv_file_paths(train_folder)
        csv_paths_dev = get_csv_file_paths(dev_folder)
        csv_paths_test = get_csv_file_paths(test_folder)

        # Get Arguments
        ctxt_size = proj.ctxt_size
        pred_size = proj.pred_size

        # Process Data
        if subset == 'training':
            csv_paths = csv_paths_train
        elif subset == 'validation':
            csv_paths = csv_paths_dev
        else:
            csv_paths = csv_paths_test
        ampro_data = []
        ampro_labels = []
        if subset == 'testing':
            ctxt_size=1
        for path in csv_paths:
            _, data, labels = collect_cps_data(path, ctxt_size, pred_size)
            ampro_data.append(data)
            ampro_labels.append(labels)

        ampro_data = np.concatenate(ampro_data, axis=0)
        ampro_labels = np.concatenate(ampro_labels, axis=0)

        # Convert data to PyTorch Tensors
        ampro_data = torch.Tensor(ampro_data).float()
        ampro_labels = torch.Tensor(ampro_labels).float()

        # Update Input size
        proj.update_attr('inp_size', ampro_data.size(-1))

        # Normalize data
        # self.mean = torch.mean(ampro_data.reshape(ampro_data.size(0) * ampro_data.size(1), -1), 0)
        # self.std = torch.std(ampro_data.reshape(ampro_data.size(0) * ampro_data.size(1), -1), 0)
        # if proj.normalize_features:
        #     if subset == 'training':
        #         ampro_data = (ampro_data - self.mean) / self.std
        #     else:
        #         ampro_data = (ampro_data - mean) / std

        # Update arguments
        proj.update_attr('inp_size', ampro_data.size(-1))
        proj.update_attr('num_classes', ampro_labels.size(-1))

        self.data = ampro_data
        self.labels = ampro_labels

        self.args_to_abb = {
            'seed': 'S',
            'rnn_size': 'H',
            'rnn_type': 'T',
            'rnn_layers': 'L',
            'num_classes': 'C',
            'ctxt_size': 'CT',
            'pred_size': 'PD',
            'qa': 'QA',
            'aqi': 'AQI',
            'aqf': 'AQF',
            'qw': 'QW',
            'wqi': 'WQI',
            'wqf': 'WQF',
        }

    def __len__(self):
        'Total number of samples'
        return self.data.size(0)  # The first dimention of the data tensor

    def __getitem__(self, idx):
        'Get one sample from the dataset using an index'
        X = self.data[idx, ...]
        y = self.labels[idx, ...]

        return X, y


def get_csv_file_paths(folder_path):
    csv_file_paths = []
    search_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(search_pattern)

    for file_path in csv_files:
        absolute_path = os.path.abspath(file_path)
        csv_file_paths.append(absolute_path)

    return csv_file_paths