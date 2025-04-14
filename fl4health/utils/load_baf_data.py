import random
import warnings
import os
from collections.abc import Callable
from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST

from fl4health.utils.dataset import TensorDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.msd_dataset_sources import get_msd_dataset_enum, msd_md5_hashes, msd_urls
from fl4health.utils.sampler import LabelBasedSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

with warnings.catch_warnings():
    # ignoring some annoying scipy deprecation warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from monai.apps.utils import download_and_extract


class ToNumpy:
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy()

class DataFrameDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, label_column: str):
        """
        Args:
            dataframe (pd.DataFrame): Input data in pandas DataFrame format.
            label_column (str): Name of the column to be used as the labels.
        """
        self.features = dataframe.drop(label_column, axis=1).values
        self.labels = dataframe[label_column].values

        # Convert features and labels to torch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def preprocess_with_labelencoder(df: pd.DataFrame, col_label: str):
    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(include=["object", "category"]).columns
    numerical_features = df.select_dtypes(include=["number"]).columns

    categorical_features = [
        features for features in categorical_features if features != col_label
    ]
    numerical_features = [
        features for features in numerical_features if features != col_label
    ]

    # Initialize dictionaries to store the encoders and scaler
    label_encoders = {}
    scaler = StandardScaler()

    # Encode categorical features using LabelEncoder
    for col in categorical_features:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Scale numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, label_encoders, scaler

def preprocess_baf_data(data:pd.DataFrame):
    COL_DF_LABEL_FRAUD = "fraud_bool"
    COL_BANK_MONTHS_COUNT = "bank_months_count"
    COL_PREV_ADDRESS_MONTHS_COUNT = "prev_address_months_count"
    COL_VELOCITY_4W = "velocity_4w"

    df = data.copy()

    cols_missing = [
        'current_address_months_count',
        'session_length_in_minutes',
        'device_distinct_emails_8w',
        'intended_balcon_amount'
    ]

    df = df.drop(columns=[
        COL_BANK_MONTHS_COUNT, 
        COL_PREV_ADDRESS_MONTHS_COUNT,
        COL_VELOCITY_4W
        ]
    )

    cols_missing = [
        'current_address_months_count',
        'session_length_in_minutes',
        'device_distinct_emails_8w',
        'intended_balcon_amount'
    ]

    df[cols_missing] = df[cols_missing].replace(-1, np.nan)
    df= df.dropna().sample(1000)

    df_preprocessed_nn, label_encoder, sclarer = preprocess_with_labelencoder(
    df=df, 
    col_label=COL_DF_LABEL_FRAUD)

    print(df_preprocessed_nn.shape)

    return df_preprocessed_nn, COL_DF_LABEL_FRAUD


def get_processed_baf_dataset(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:

    baf_dataset = pd.read_csv(f"{data_dir}.csv")

    processed_baf_dataset, COL_DF_LABEL_FRAUD = preprocess_baf_data(baf_dataset)

    return processed_baf_dataset, COL_DF_LABEL_FRAUD



def split_data_and_targets(
    data, target: str, val_size: float = 0.50, test_size: float = 0.30, hash_key: int | None = None ):

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=data[target],
    )

    test_df, val_df = train_test_split(
        test_df,
        test_size=val_size,
        shuffle=True,
        random_state=42,
        stratify=test_df[target],
    )

    # Separate features and labels for SMOTE
    majority_class = train_df[train_df[target] == 0]
    minority_class = train_df[train_df[target] == 1]

    # Oversample the minority class by duplicating rows
    minority_oversampled = minority_class.sample(
        n=len(majority_class), replace=True, random_state=42
    )

    # Combine and shuffle
    train_df_balanced = pd.concat([majority_class, minority_oversampled])
    train_df_balanced = train_df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(train_df_balanced.shape)

    return train_df_balanced, val_df, test_df

def get_train_and_val_baf_datasets(
    data_dir: Path,
    val_size: float = 0.50,
    test_size: float = 0.30,
    hash_key: int | None = None,
) -> tuple[TensorDataset, TensorDataset]:
    data, target = get_processed_baf_dataset(data_dir)

    train_df_balanced, val_df, test_df = split_data_and_targets(
        data, target, val_size, test_size, hash_key
    )

    train_dataset = DataFrameDataset(train_df_balanced, label_column=target)
    val_dataset = DataFrameDataset(val_df, label_column=target)
    test_dataset = DataFrameDataset(test_df, label_column=target)

    return train_dataset, val_dataset, test_dataset


def load_baf_data(
    data_dir: Path,
    val_size: int = 0.5,
    test_size: int = 0.3,
    batch_size: int = 32,
    num_workers: int = os.cpu_count(),
    sampler: LabelBasedSampler | None = None,
    hash_key: int | None = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """
    Load BAF Dataset (training and validation set).

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

 
    train_dataset, val_dataset, test_dataset = get_train_and_val_baf_datasets(
        data_dir, val_size, test_size, hash_key
    )

    if sampler is not None:
        training_set = sampler.subsample(training_set)
        validation_set = sampler.subsample(validation_set)

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True
        )
    validation_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )


    num_examples = {"train_set": len(train_dataset), "validation_set": len(val_dataset)}

    return train_loader, validation_loader, num_examples

def load_baf_test_data(
    data_dir: Path,
    batch_size: int,
    sampler: LabelBasedSampler | None = None,
    transform: Callable | None = None,
    num_workers=os.cpu_count(),
    shuffle=False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
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

    _ , _,  test_dataset = get_train_and_val_baf_datasets(data_dir)

    
    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    num_examples = {"test_set": len(test_dataset)}
    return test_loader, num_examples
