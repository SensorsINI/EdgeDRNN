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
    df = pd.read_csv(filepath)

    # Collect data
    ishumanswing = df.ishumanswing.to_numpy()
    position_actual_ankle = df.position_actual_ankle.to_numpy()
    position_actual_knee = df.position_actual_knee.to_numpy()
    velocity_actual_ankle = df.velocity_actual_ankle.to_numpy()
    velocity_actual_knee = df.velocity_actual_knee.to_numpy()
    # torque_actual_ankle = df.torque_actual_ankle.to_numpy()
    # torque_actual_knee = df.torque_actual_knee.to_numpy()
    # position_desired_ankle = df.position_desired_ankle.to_numpy()
    position_desired_knee = df.position_desired_knee.to_numpy()
    velocity_desired_knee = df.velocity_desired_knee.to_numpy()
    position_error_knee = position_actual_knee - position_desired_knee
    velocity_error_knee = velocity_actual_knee - velocity_desired_knee

    # Train Data
    train_data = []
    train_data.append(ishumanswing)  # Binary Flag - Whether the leg is swinging
    train_data.append(position_error_knee)  # Knee position error
    train_data.append(velocity_error_knee)  # Knee velocity
    train_data.append(
        position_actual_ankle)  # Equivalent to the ankle position error since the desired position is always 0
    train_data.append(
        velocity_actual_ankle)  # Equivalent to the ankle velocity error since the desired velocity is always 0
    train_data = np.vstack(train_data).transpose()

    # SPI Data for MiniZed-BBB Debugging
    spi_data = []
    spi_data.append(ishumanswing)
    spi_data.append(position_actual_knee)
    spi_data.append(position_desired_knee)
    spi_data.append(velocity_actual_knee)
    spi_data.append(velocity_desired_knee)
    spi_data.append(position_actual_ankle)
    spi_data.append(velocity_actual_ankle)

    # Collect label
    raw_labels = []
    raw_labels.append(df.torque_desired_ankle.to_numpy())
    raw_labels.append(df.torque_desired_knee.to_numpy())
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
        self.batch_first = proj.batch_first
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
        if not self.batch_first:
            X = X.transpose(0, 1)
        return X, y


def get_csv_file_paths(folder_path):
    csv_file_paths = []
    search_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(search_pattern)

    for file_path in csv_files:
        absolute_path = os.path.abspath(file_path)
        csv_file_paths.append(absolute_path)

    return csv_file_paths
