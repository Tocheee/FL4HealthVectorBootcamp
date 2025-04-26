import numpy as np
import pandas as pd
import torch

from collections import Counter
from torch.utils.data import Dataset

def compute_class_counts(dataset: Dataset) -> dict[int, int]:
    """
    Compute class distribution from a PyTorch Dataset.

    Args:
        dataset (Dataset): A PyTorch Dataset where labels are in the second element of each item.

    Returns:
        dict[int, int]: Dictionary of class counts (label -> count).
    """
    # Extract labels from dataset and convert to ints
    labels = [int(dataset[i][1].item()) for i in range(len(dataset))]
    return dict(Counter(labels))
