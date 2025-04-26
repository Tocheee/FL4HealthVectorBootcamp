import random
import warnings
from collections.abc import Callable
from logging import INFO
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

from fl4health.utils.dataset import TensorDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.msd_dataset_sources import get_msd_dataset_enum, msd_md5_hashes, msd_urls
from fl4health.utils.sampler import LabelBasedSampler

with warnings.catch_warnings():
    # ignoring some annoying scipy deprecation warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from monai.apps.utils import download_and_extract


class ToNumpy:
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy()


def split_data_and_targets(
    data: torch.Tensor, targets: torch.Tensor, validation_proportion: float = 0.2, hash_key: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    total_size = data.shape[0]
    train_size = int(total_size * (1 - validation_proportion))
    if hash_key is not None:
        random.seed(hash_key)
    train_indices = random.sample(range(total_size), train_size)
    val_indices = [i for i in range(total_size) if i not in train_indices]
    train_data, train_targets = data[train_indices], targets[train_indices]
    val_data, val_targets = data[val_indices], targets[val_indices]
    return train_data, train_targets, val_data, val_targets


def get_mnist_data_and_target_tensors(data_dir: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    mnist_dataset = MNIST(data_dir, train=train, download=True)
    data = torch.Tensor(mnist_dataset.data)
    targets = torch.Tensor(mnist_dataset.targets).long()
    return data, targets


def get_train_and_val_mnist_datasets(
    data_dir: Path,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    validation_proportion: float = 0.2,
    hash_key: int | None = None,
) -> tuple[TensorDataset, TensorDataset]:
    data, targets = get_mnist_data_and_target_tensors(data_dir, True)

    train_data, train_targets, val_data, val_targets = split_data_and_targets(
        data, targets, validation_proportion, hash_key
    )

    training_set = TensorDataset(train_data, train_targets, transform=transform, target_transform=target_transform)
    validation_set = TensorDataset(val_data, val_targets, transform=transform, target_transform=target_transform)
    return training_set, validation_set


def load_mnist_data(
    data_dir: Path,
    batch_size: int,
    sampler: LabelBasedSampler | None = None,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    dataset_converter: DatasetConverter | None = None,
    validation_proportion: float = 0.2,
    hash_key: int | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Load MNIST Dataset (training and validation set).

    Args:
        data_dir (Path): The path to the MNIST dataset locally. Dataset is downloaded to this location if it does
            not already exist.
        batch_size (int): The batch size to use for the train and validation dataloader.
        sampler (LabelBasedSampler | None): Optional sampler to subsample dataset based on labels.
        transform (Callable | None): Optional transform to be applied to input samples.
        target_transform (Callable | None): Optional transform to be applied to targets.
        dataset_converter (DatasetConverter | None): Optional dataset converter used to convert the input and/or
            target of train and validation dataset.
        validation_proportion (float): A float between 0 and 1 specifying the proportion of samples
            to allocate to the validation dataset. Defaults to 0.2.
        hash_key (int | None): Optional hash key to create a reproducible split for train and validation
            datasets.

    Returns:
        tuple[DataLoader, DataLoader, dict[str, int]]: The train data loader, validation data loader and a dictionary
        with the sample counts of datasets underpinning the respective data loaders.
    """
    log(INFO, f"Data directory: {str(data_dir)}")

    if transform is None:
        transform = transforms.Compose(
            [
                ToNumpy(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
    training_set, validation_set = get_train_and_val_mnist_datasets(
        data_dir, transform, target_transform, validation_proportion, hash_key
    )

    if sampler is not None:
        training_set = sampler.subsample(training_set)
        validation_set = sampler.subsample(validation_set)

    if dataset_converter is not None:
        training_set = dataset_converter.convert_dataset(training_set)
        validation_set = dataset_converter.convert_dataset(validation_set)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)

    num_examples = {"train_set": len(training_set), "validation_set": len(validation_set)}
    return train_loader, validation_loader, num_examples


def load_mnist_test_data(
    data_dir: Path,
    batch_size: int,
    sampler: LabelBasedSampler | None = None,
    transform: Callable | None = None,
) -> tuple[DataLoader, dict[str, int]]:
    """
    Load MNIST Test Dataset.

    Args:
        data_dir (Path): The path to the MNIST dataset locally. Dataset is downloaded to this location if it does not
            already exist.
        batch_size (int): The batch size to use for the test dataloader.
        sampler (LabelBasedSampler | None): Optional sampler to subsample dataset based on labels.
        transform (Callable | None): Optional transform to be applied to input samples.

    Returns:
        tuple[DataLoader, dict[str, int]]: The test data loader and a dictionary containing the sample count
            of the test dataset.
    """
    log(INFO, f"Data directory: {str(data_dir)}")

    if transform is None:
        transform = transforms.Compose(
            [
                ToNumpy(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

    data, targets = get_mnist_data_and_target_tensors(data_dir, False)
    evaluation_set = TensorDataset(data, targets, transform)

    if sampler is not None:
        evaluation_set = sampler.subsample(evaluation_set)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}
    return evaluation_loader, num_examples


def get_cifar10_data_and_target_tensors(data_dir: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    cifar_dataset = CIFAR10(data_dir, train=train, download=True)
    data = torch.Tensor(cifar_dataset.data)
    targets = torch.Tensor(cifar_dataset.targets).long()
    return data, targets


def get_train_and_val_cifar10_datasets(
    data_dir: Path,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    validation_proportion: float = 0.2,
    hash_key: int | None = None,
) -> tuple[TensorDataset, TensorDataset]:
    data, targets = get_cifar10_data_and_target_tensors(data_dir, True)

    train_data, train_targets, val_data, val_targets = split_data_and_targets(
        data, targets, validation_proportion, hash_key
    )

    training_set = TensorDataset(train_data, train_targets, transform=transform, target_transform=target_transform)
    validation_set = TensorDataset(val_data, val_targets, transform=transform, target_transform=target_transform)

    return training_set, validation_set


def load_cifar10_data(
    data_dir: Path,
    batch_size: int,
    sampler: LabelBasedSampler | None = None,
    validation_proportion: float = 0.2,
    hash_key: int | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Load CIFAR10 Dataset (training and validation set).

    Args:
        data_dir (Path): The path to the CIFAR10 dataset locally. Dataset is downloaded to this location if it does
            not already exist.
        batch_size (int): The batch size to use for the train and validation dataloader.
        sampler (LabelBasedSampler | None): Optional sampler to subsample dataset based on labels.
        validation_proportion (float): A float between 0 and 1 specifying the proportion of samples to allocate to the
            validation dataset. Defaults to 0.2.
        hash_key (int | None): Optional hash key to create a reproducible split for train and validation
            datasets.

    Returns:
        tuple[DataLoader, DataLoader, dict[str, int]]: The train data loader, validation data loader and a dictionary
        with the sample counts of datasets underpinning the respective data loaders.
    """
    log(INFO, f"Data directory: {str(data_dir)}")

    transform = transforms.Compose(
        [
            ToNumpy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    training_set, validation_set = get_train_and_val_cifar10_datasets(
        data_dir, transform, None, validation_proportion, hash_key
    )

    if sampler is not None:
        training_set = sampler.subsample(training_set)
        validation_set = sampler.subsample(validation_set)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(training_set),
        "validation_set": len(validation_set),
    }
    return train_loader, validation_loader, num_examples


def load_cifar10_test_data(
    data_dir: Path, batch_size: int, sampler: LabelBasedSampler | None = None
) -> tuple[DataLoader, dict[str, int]]:
    """
    Load CIFAR10 Test Dataset.

    Args:
        data_dir (Path): The path to the CIFAR10 dataset locally. Dataset is downloaded to this location if it does
            not already exist.
        batch_size (int): The batch size to use for the test dataloader.
        sampler (LabelBasedSampler | None): Optional sampler to subsample dataset based on labels.

    Returns:
        tuple[DataLoader, dict[str, int]]: The test data loader and a dictionary containing the sample count of the
        test dataset.
    """
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
            ToNumpy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    data, targets = get_cifar10_data_and_target_tensors(data_dir, False)
    evaluation_set = TensorDataset(data, targets, transform)

    if sampler is not None:
        evaluation_set = sampler.subsample(evaluation_set)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}
    return evaluation_loader, num_examples


def load_msd_dataset(data_path: str, msd_dataset_name: str) -> None:
    """
    Downloads and extracts one of the 10 Medical Segmentation Decathelon (MSD) datasets.

    Args:
        data_path (str): Path to the folder in which to extract the dataset. The data itself will be in a subfolder
            named after the dataset, not in the ``data_path`` directory itself. The name of the folder will be the
            name of the dataset as defined by the values of the ``MsdDataset`` enum returned by
            ``get_msd_dataset_enum``
        msd_dataset_name (str): One of the 10 msd datasets
    """
    msd_enum = get_msd_dataset_enum(msd_dataset_name)
    msd_hash = msd_md5_hashes[msd_enum]
    url = msd_urls[msd_enum]
    download_and_extract(url=url, output_dir=data_path, hash_val=msd_hash, hash_type="md5", progress=True)




import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Tuple
import joblib
from torch.utils.data import WeightedRandomSampler
from sklearn.utils import resample

            
class TabularScaler:
    def __init__(self, numeric_features: List[str], categorical_features: List[str]) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaler = None
        self.encoder = None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Fill missing columns before fitting
        for col in self.numeric_features:
            if col not in X.columns:
                X[col] = 0.0
        for col in self.categorical_features:
            if col not in X.columns:
                X[col] = "unknown"

        # Reorder to ensure consistent order
        X = X[self.numeric_features + self.categorical_features]

        numeric_data_scaled = self.scaler.fit_transform(X[self.numeric_features])
        categorical_data_encoded = self.encoder.fit_transform(X[self.categorical_features])
        return np.hstack((numeric_data_scaled, categorical_data_encoded))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Fill missing numeric features with 0.0
        for col in self.numeric_features:
            if col not in X.columns:
                X[col] = 0.0

        # Fill missing categorical features with "unknown"
        for col in self.categorical_features:
            if col not in X.columns:
                X[col] = "unknown"

        # Reorder columns to match the fit order
        X = X[self.numeric_features + self.categorical_features]

        numeric_data_scaled = self.scaler.transform(X[self.numeric_features])
        categorical_data_encoded = self.encoder.transform(X[self.categorical_features])
        return np.hstack((numeric_data_scaled, categorical_data_encoded))


class DataPrep:
    def __init__(self, data_file_path: Path, scaler_path: Path, batch_size: int):
        self.data_file_path = data_file_path
        self.scaler = joblib.load(scaler_path)
        self.batch_size = batch_size

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.input_dim = None
        self.num_examples = {}

        self._prepare_data()

    def _prepare_data(self):
        print(f"\nLoading data from: {self.data_file_path}\n")
        df = pd.read_csv(self.data_file_path)

        # Drop redundant columns
        df = df.drop(columns=[
            "bank_months_count",
            "prev_address_months_count",
            "velocity_4w"
        ])

        # Handle missing values
        cols_missing = [
            'current_address_months_count',
            'session_length_in_minutes',
            'device_distinct_emails_8w',
            'intended_balcon_amount'
        ]
        df[cols_missing] = df[cols_missing].replace(-1, np.nan)
        df = df.dropna()

        target_col = "fraud_bool"
        y = df[target_col].values
        X = df.drop(columns=[target_col])

        # Train/val/test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

        # Custom Oversampling (1:5 ratio with Gaussian noise)
        df_train = X_train.copy()
        df_train['label'] = y_train

        majority = df_train[df_train.label == 0]
        minority = df_train[df_train.label == 1]

        # Determine new minority size (capped to 1:5)
        minority_upsampled = resample(minority, replace=True, n_samples=int(len(majority) * 0.20), random_state=42)

        # Add Gaussian noise to numeric features
        numeric_cols = self.scaler.numeric_features
        minority_augmented = minority_upsampled.copy()
        noise = np.random.normal(loc=0.0, scale=0.01, size=minority_augmented[numeric_cols].shape)
        minority_augmented[numeric_cols] += noise

        # Combine and shuffle
        df_balanced = pd.concat([majority, minority_augmented])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        y_train_bal = df_balanced['label'].values
        X_train_bal = df_balanced.drop(columns=['label'])

        # Scale using the global scaler
        X_train_scaled = self.scaler.transform(X_train_bal)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        self.input_dim = X_train_scaled.shape[1]

        # Convert to tensors
        def make_loader(X_data, y_data, use_sampler=False, shuffle=True):
            X_tensor = torch.tensor(X_data, dtype=torch.float32)
            y_tensor = torch.tensor(y_data, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)

            if use_sampler:
                # Compute class weights
                class_sample_count = np.array([len(np.where(y_data == t)[0]) for t in np.unique(y_data)])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[int(t)] for t in y_data])
                samples_weight = torch.from_numpy(samples_weight).double()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
            else:
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train_loader = make_loader(X_train_scaled, y_train_bal)
        self.val_loader = make_loader(X_val_scaled, y_val, shuffle = False)
        self.test_loader = make_loader(X_test_scaled, y_test, shuffle = False)

        self.num_examples = {
            "train_set": len(y_train),
            "val_set": len(y_val),
            "test_set": len(y_test),
        }

    def get_train_val_loaders(self):
        return self.train_loader, self.val_loader

    def get_test_loader(self):
        return self.test_loader

    def get_input_dim(self):
        return self.input_dim

    def get_num_examples(self):
        return self.num_examples
