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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from pathlib import Path
from typing import List, Tuple

class TabularScaler:
    def __init__(self, numeric_features: List[str], categorical_features: List[str]) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        # Apply standard scaling to numeric columns
        numeric_data = X[self.numeric_features]
        numeric_data_scaled = self.scaler.fit_transform(numeric_data)

        # Apply one-hot encoding to categorical columns
        categorical_data = X[self.categorical_features]
        categorical_data_encoded = self.encoder.fit_transform(categorical_data)

        # Combine numeric and categorical data
        transformed_data = np.hstack((numeric_data_scaled, categorical_data_encoded))
        return transformed_data

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Apply standard scaling to numeric columns
        numeric_data = X[self.numeric_features]
        numeric_data_scaled = self.scaler.transform(numeric_data)

        # Apply one-hot encoding to categorical columns
        categorical_data = X[self.categorical_features]
        categorical_data_encoded = self.encoder.transform(categorical_data)

        # Combine numeric and categorical data
        transformed_data = np.hstack((numeric_data_scaled, categorical_data_encoded))
        return transformed_data


def load_data(data_dir: Path, batch_size: int) -> Tuple[DataLoader, DataLoader, dict[str, int]]:
    # Get the list of CSV files in the data directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if len(all_files) == 0:
        raise ValueError(f"No CSV files found in the directory: {data_dir}")

    # Automatically select the file (you can still map it to a client index if needed)
    file_name = all_files[0]  # Adjust based on your strategy
    data_path = data_dir / file_name

    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_path)
    if 'device_distinct_emails' in df.columns:
        df = df.drop(columns = 'device_distinct_emails')

    # Set the target column and input features
    target_col = "fraud_bool"  # Target column should match your dataset
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    # Columns by type
    numeric_cols = [
        "income", "name_email_similarity", "prev_address_months_count",
        "current_address_months_count", "customer_age", "days_since_request",
        "intended_balcon_amount", "zip_count_4w", "velocity_6h",
        "velocity_24h", "velocity_4w", "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w", "credit_risk_score",
        "bank_months_count", "proposed_credit_limit", "session_length_in_minutes",
        "device_fraud_count", "month" #device_distinct_emails
    ]

    categorical_cols = [
        "payment_type", "employment_status", "housing_status",
        "source", "device_os"
    ]

    binary_cols = [
        "email_is_free", "phone_home_valid", "phone_mobile_valid",
        "has_other_cards", "foreign_request", "keep_alive_session"
    ]

    # Filter the columns
    all_input_cols = numeric_cols + categorical_cols + binary_cols
    X = X[all_input_cols]

    # Train/test split
    n_samples = len(df)
    split_index = int(n_samples * 0.8)

    X_train, y_train = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]

    # Instantiate the TabularScaler with the numeric and categorical columns
    scaler = TabularScaler(numeric_features=numeric_cols, categorical_features=categorical_cols)

    # Apply scaler and encoder
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled).float()
    y_train_tensor = torch.tensor(y_train).float()

    X_val_tensor = torch.tensor(X_val_scaled).float()
    y_val_tensor = torch.tensor(y_val).float()

    # Wrap into TensorDataset and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_examples = {"train_set": len(train_dataset), "validation_set": len(val_dataset)}

    return train_loader, val_loader, num_examples
    