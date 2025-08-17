"""
EEG Data Processing Pipeline
Author: dddyuu
Date: 2025-08-17

This module provides functionality for loading and processing EEG data from DTU and KUL datasets,
including signal preprocessing, SPD matrix computation, and data augmentation.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import scipy.io as sio
from dotmap import DotMap


class EEGDataset(Dataset):
    """Custom dataset class for EEG data with SPD matrices"""

    def __init__(self, data, adj, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.adj = torch.tensor(adj, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.adj[idx], self.labels[idx]


class Signal2SPD(nn.Module):
    """Convert EEG signal epochs to SPD (Symmetric Positive Definite) matrices"""

    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.device = torch.device('cpu')

    def forward(self, x):
        """
        Convert signal to SPD matrix via covariance computation

        Args:
            x: Input signal [batch_size, channels, time_points]

        Returns:
            SPD covariance matrices [batch_size, channels, channels]
        """
        x = x.squeeze()

        # Center the data
        mean = x.mean(axis=-1, keepdim=True)
        x_centered = x - mean

        # Compute covariance matrix
        cov = torch.bmm(x_centered, x_centered.transpose(-1, -2))
        cov = cov / (x.shape[-1] - 1)

        # Normalize by trace
        trace = torch.diagonal(cov, dim1=-1, dim2=-2).sum(-1, keepdim=True).unsqueeze(-1)
        cov = cov / trace

        # Add regularization for numerical stability
        identity = torch.eye(cov.shape[-1], device=self.device).repeat(x.shape[0], 1, 1)
        cov = cov + self.epsilon * identity

        return cov


class EEG2Riemann(nn.Module):
    """Convert EEG signals to Riemannian manifold representation"""

    def __init__(self, n_epochs=1):
        super().__init__()
        self.n_epochs = n_epochs
        self.signal2spd = Signal2SPD()

    def _compute_patch_lengths(self, total_length, n_epochs):
        """Compute patch lengths for splitting time series"""
        base_length = total_length // n_epochs
        patch_lengths = [base_length] * n_epochs

        # Distribute remaining samples
        remainder = total_length - base_length * n_epochs
        for i in range(remainder):
            patch_lengths[i] += 1

        return patch_lengths

    def forward(self, x):
        """
        Convert EEG signal to SPD representation

        Args:
            x: EEG signal [batch_size, channels, time_points]

        Returns:
            SPD matrices [batch_size, n_epochs, channels, channels]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        patch_lengths = self._compute_patch_lengths(x.shape[-1], self.n_epochs)
        patches = torch.split(x, patch_lengths, dim=-1)

        spd_matrices = []
        for patch in patches:
            spd_matrices.append(self.signal2spd(patch))

        result = torch.stack(spd_matrices, dim=1)
        return result


class DTUDataProcessor:
    """Data processor for DTU dataset"""

    def __init__(self, data_path="./DTU/DATA_preproc", fs=64, overlap=0.6):
        self.data_path = data_path
        self.fs = fs
        self.overlap = overlap
        self.eeg_channels = 64
        self.n_trials = 60

    def load_subject_data(self, subject_name, time_len=1):
        """Load and preprocess data for a single subject"""
        file_path = os.path.join(self.data_path, f"{subject_name}_data_preproc.mat")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load MATLAB file
        mat_data = loadmat(file_path)['data']

        # Extract EEG data and labels
        eeg_trials = []
        labels = []

        for trial_idx in range(self.n_trials):
            # Extract EEG data (shape: [time_points, channels])
            trial_data = mat_data[0, 0]['eeg'][0, trial_idx][:, :self.eeg_channels]
            eeg_trials.append(trial_data)

            # Extract labels
            label = mat_data[0, 0]['event']['eeg'].item()[0]['value'][trial_idx][0][0]
            # Convert labels: 1->0, 2->1
            labels.append(0 if label == 1 else 1)

        return np.array(eeg_trials), np.array(labels)

    def apply_sliding_window(self, eeg_data, labels, time_len):
        """Apply sliding window segmentation"""
        window_size = math.ceil(self.fs * time_len)
        stride = int(window_size * (1 - self.overlap))

        windowed_data = []
        windowed_labels = []

        for trial_data, label in zip(eeg_data, labels):
            for start_idx in range(0, trial_data.shape[0] - window_size + 1, stride):
                window = trial_data[start_idx:start_idx + window_size, :]
                windowed_data.append(window)
                windowed_labels.append(label)

        return np.array(windowed_data), np.array(windowed_labels)

    def process_subject(self, subject_name, time_len=1):
        """Complete processing pipeline for DTU subject"""
        print(f"Processing DTU subject: {subject_name}")

        # Load raw data
        eeg_data, labels = self.load_subject_data(subject_name, time_len)
        print(f"Loaded {len(eeg_data)} trials with shape {eeg_data.shape}")

        # Apply sliding window
        windowed_data, windowed_labels = self.apply_sliding_window(eeg_data, labels, time_len)
        print(f"Created {len(windowed_data)} windows with shape {windowed_data.shape}")

        # Transpose to [samples, channels, time]
        windowed_data = windowed_data.transpose(0, 2, 1)

        # Split into train/valid/test
        n_samples = len(windowed_labels)
        n_test = int(n_samples * 0.1)
        n_valid = n_test
        n_train = n_samples - n_test - n_valid

        # Shuffle data
        indices = np.random.permutation(n_samples)
        windowed_data = windowed_data[indices]
        windowed_labels = windowed_labels[indices]

        # Split datasets
        train_data = windowed_data[:n_train]
        valid_data = windowed_data[n_train:n_train + n_valid]
        test_data = windowed_data[n_train + n_valid:]

        train_labels = windowed_labels[:n_train]
        valid_labels = windowed_labels[n_train:n_train + n_valid]
        test_labels = windowed_labels[n_train + n_valid:]

        # Generate SPD matrices
        e2r_model = EEG2Riemann(n_epochs=1)

        train_spd = e2r_model(train_data).squeeze(1)
        valid_spd = e2r_model(valid_data).squeeze(1)
        test_spd = e2r_model(test_data).squeeze(1)

        return {
            'train': (train_data, train_spd, train_labels),
            'valid': (valid_data, valid_spd, valid_labels),
            'test': (test_data, test_spd, test_labels)
        }


class KULDataProcessor:
    """Data processor for KUL dataset"""

    def __init__(self, data_path="./KUL/preprocessed_data", fs=128, overlap=0.5):
        self.data_path = data_path
        self.fs = fs
        self.overlap = overlap
        self.eeg_channels = 64
        self.n_trials_per_subject = 8

    def load_subject_data(self, subject_name):
        """Load and preprocess data for a single KUL subject"""
        file_path = os.path.join(self.data_path, f"{subject_name}.mat")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load MATLAB file
        mat_data = sio.loadmat(file_path)
        trials = mat_data['preproc_trials']

        eeg_trials = []
        labels = []

        for trial_idx in range(self.n_trials_per_subject):
            trial_info = trials[0][trial_idx]

            # Extract EEG data
            raw_data = trial_info['RawData']
            trial_data = raw_data[0][0]['EegData'].item()

            # Truncate to standard length
            trial_data = trial_data[:49792, :]
            eeg_trials.append(trial_data)

            # Extract and convert labels
            label = trial_info['attended_ear'].item()
            labels.append(1 if label == 'R' else 0)

        return np.array(eeg_trials), np.array(labels)

    def apply_sliding_window(self, eeg_data, labels, time_len):
        """Apply sliding window segmentation"""
        window_size = math.ceil(self.fs * time_len)
        stride = int(window_size * (1 - self.overlap))

        windowed_data = []
        windowed_labels = []

        for trial_data, label in zip(eeg_data, labels):
            for start_idx in range(0, trial_data.shape[0] - window_size + 1, stride):
                window = trial_data[start_idx:start_idx + window_size, :]
                windowed_data.append(window)
                windowed_labels.append(label)

        return np.array(windowed_data), np.array(windowed_labels)

    def process_subject(self, subject_name, time_len=1):
        """Complete processing pipeline for KUL subject"""
        print(f"Processing KUL subject: {subject_name}")

        # Load raw data
        eeg_data, labels = self.load_subject_data(subject_name)
        print(f"Loaded {len(eeg_data)} trials")

        # Apply sliding window
        windowed_data, windowed_labels = self.apply_sliding_window(eeg_data, labels, time_len)
        print(f"Created {len(windowed_data)} windows with shape {windowed_data.shape}")

        # Transpose to [samples, channels, time]
        windowed_data = windowed_data.transpose(0, 2, 1)

        # Split into train/valid/test
        n_samples = len(windowed_labels)
        n_test = int(n_samples * 0.1)
        n_valid = n_test
        n_train = n_samples - n_test - n_valid

        # Shuffle data
        indices = np.random.permutation(n_samples)
        windowed_data = windowed_data[indices]
        windowed_labels = windowed_labels[indices]

        # Split datasets
        train_data = windowed_data[:n_train]
        valid_data = windowed_data[n_train:n_train + n_valid]
        test_data = windowed_data[n_train + n_valid:]

        train_labels = windowed_labels[:n_train]
        valid_labels = windowed_labels[n_train:n_train + n_valid]
        test_labels = windowed_labels[n_train + n_valid:]

        # Generate SPD matrices
        e2r_model = EEG2Riemann(n_epochs=1)

        train_spd = e2r_model(train_data).squeeze(1)
        valid_spd = e2r_model(valid_data).squeeze(1)
        test_spd = e2r_model(test_data).squeeze(1)

        return {
            'train': (train_data, train_spd, train_labels),
            'valid': (valid_data, valid_spd, valid_labels),
            'test': (test_data, test_spd, test_labels)
        }


def create_data_loaders(processed_data, batch_size=32):
    """Create PyTorch data loaders from processed data"""
    datasets = {}
    loaders = {}

    for split in ['train', 'valid', 'test']:
        data, spd, labels = processed_data[split]
        datasets[split] = EEGDataset(data, spd, labels)

        shuffle = (split == 'train')
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
            num_workers=2
        )

    return loaders['train'], loaders['valid'], loaders['test']


def get_data(subject_name, time_len=1, dataset_name="KUL", **kwargs):
    """
    Main function to get data loaders for a subject

    Args:
        subject_name: Subject identifier (e.g., "S1")
        time_len: Time window length in seconds
        dataset_name: Dataset name ("KUL" or "DTU")
        **kwargs: Additional arguments for data paths

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    if dataset_name.upper() == "DTU":
        data_path = kwargs.get('data_path', './DTU/DATA_preproc')
        processor = DTUDataProcessor(data_path=data_path)
    elif dataset_name.upper() == "KUL":
        data_path = kwargs.get('data_path', './KUL/preprocessed_data')
        processor = KULDataProcessor(data_path=data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Process subject data
    processed_data = processor.process_subject(subject_name, time_len)

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        processed_data,
        batch_size=kwargs.get('batch_size', 32)
    )

    # Optional: Save test data for later analysis
    save_path = kwargs.get('save_path', './saved_data')
    if save_path and os.path.exists(os.path.dirname(save_path)):
        os.makedirs(save_path, exist_ok=True)
        test_data, test_spd, test_labels = processed_data['test']
        np.save(f'{save_path}/test_data_{subject_name}.npy', test_data)
        np.save(f'{save_path}/test_spd_{subject_name}.npy', test_spd)
        np.save(f'{save_path}/test_labels_{subject_name}.npy', test_labels)

    print(f"Data loaders created for {subject_name}")
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Valid: {len(valid_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    return train_loader, valid_loader, test_loader


# Alias functions for backward compatibility
def getData(subject_name, time_len=1, dataset_name="KUL"):
    """Legacy function name"""
    return get_data(subject_name, time_len, dataset_name)


def get_DTU_data(name="S1", timelen=1, data_document_path="./DTU/DATA_preproc"):
    """Legacy DTU data loading function"""
    return get_data(name, timelen, "DTU", data_path=data_document_path)


def get_KUL_data(name="S1", time_len=1, data_document_path="./KUL/preprocessed_data"):
    """Legacy KUL data loading function"""
    return get_data(name, time_len, "KUL", data_path=data_document_path)


if __name__ == "__main__":
    # Test the data processing pipeline
    print("Testing EEG data processing pipeline...")

    try:
        # Test KUL dataset
        train_loader, valid_loader, test_loader = get_data("S1", time_len=1, dataset_name="KUL")
        print("✓ KUL data processing successful")

        # Test a batch
        for batch_data, batch_spd, batch_labels in train_loader:
            print(f"Batch shapes - Data: {batch_data.shape}, SPD: {batch_spd.shape}, Labels: {batch_labels.shape}")
            break

    except Exception as e:
        print(f"✗ Error testing data processing: {e}")

    print("Data processing pipeline test completed!")