import numpy as np
import pandas as pd
import torch

def compute_class_counts(data_path: str) -> dict[int, int]:
    """
    Compute the class distribution from the fraud detection dataset.
    Args:
        data_path (str): Path to the CSV file containing the dataset.
    
    Returns:
        dict: A dictionary with class labels as keys and their respective counts as values.
    """
    # Load the dataset
    df = pd.read_csv(data_path, index_col=False)
    
    # Assuming the target column is "fraud_bool"
    target_col = "fraud_bool"
    class_counts = df[target_col].value_counts().to_dict()

    return class_counts


def compute_class_weights(class_counts: dict[int, int]) -> torch.Tensor:
    """
    Compute the weights for each class based on their distribution.
    Args:
        class_counts (dict): A dictionary with class labels as keys and their respective counts as values.
    
    Returns:
        torch.Tensor: A tensor containing the computed class weights for each class.
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    # Compute weight for each class as the inverse of its frequency
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
    
    # Convert to tensor
    weights = torch.tensor([class_weights.get(0, 0), class_weights.get(1, 0)], dtype=torch.float)

    return weights
