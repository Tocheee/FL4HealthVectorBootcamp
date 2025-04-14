import argparse
from pathlib import Path
import os

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.mlp_classifier import MLPNet
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import DataPrep
from fl4health.utils.metrics import Accuracy, BalancedAccuracy, F1
from fl4health.utils.class_weights import compute_class_counts

class FraudClient(BasicClient):
    def __init__(self, data_file_path: str, scaler_path: str, metrics: list, device: torch.device):
        super().__init__(data_path=data_file_path, metrics=metrics, device=device)
        self.data_file_path = data_file_path
        self.scaler_path = scaler_path
        self.device = device
        self.metrics = metrics
        
        self.data_loader = DataPrep(data_file_path=Path(self.data_file_path), scaler_path=Path(self.scaler_path))
        self.input_dim = self.data_loader.get_input_dim()

        self.model = self.get_model({})
        self.initialized = True
        print(self.input_dim)     

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _, _, _ = self.data_loader.load_baf_data(batch_size=batch_size)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        _, _, test_loader, _, _ = self.data_loader.load_baf_data(batch_size=batch_size)
        return test_loader
    
    def get_criterion(self, config: Config) -> _Loss:
        counts = compute_class_counts(self.data_file_path)
        pos_weight = torch.tensor([counts[0] / counts[1]]) 
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        return MLPNet(input_dim = self.input_dim).to(self.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    args = parser.parse_args()
    
    data_path = os.environ.get("data_file_path")
    scaler_path = os.environ.get("scaler_save_path")
    print(f"Using data path: {data_path}")
    print(f"Using scaler path: {scaler_path}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    client = FraudClient(data_file_path = data_path, scaler_path = scaler_path, metrics = [BalancedAccuracy("balanced_accuracy"), F1("f1_score")], device = device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
