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
    def __init__(self, data_file_path: str, scaler_path: str, metrics: list, device: torch.device, progress_bar = True):
        super().__init__(data_path=data_file_path, metrics=metrics, device=device, progress_bar = True)

        self.data_file_path = data_file_path
        self.scaler_path = scaler_path
        self.device = device
        self.metrics = metrics

        # Instantiate new DataPrep
        self.data_loader = DataPrep(
            data_file_path=Path(self.data_file_path),
            scaler_path=Path(self.scaler_path),
            batch_size=32
        )

        # Expose required attributes to BasicClient
        self.train_loader, self.val_loader = self.data_loader.get_train_val_loaders()
        self.test_loader = self.data_loader.get_test_loader()
        self.input_dim = self.data_loader.get_input_dim()
        self.num_train_samples = len(self.train_loader.dataset)
        self.num_val_samples = len(self.val_loader.dataset)
        self.num_test_samples = len(self.test_loader.dataset)

        print("Data dimension: ",self.input_dim)
        
        self.model = self.get_model({})
        self.optimizers = {"global": self.get_optimizer({})}
        self.lr_schedulers = {}
        self.criterion = self.get_criterion({})
        self.parameter_exchanger = self.get_parameter_exchanger({})
        self.initialized = True

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        # Already loaded and stored in init
        return self.train_loader, self.val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        return self.test_loader

    def get_criterion(self, config: Config) -> _Loss:
        counts = compute_class_counts(self.data_file_path)
        pos_weight = torch.tensor([counts[0] / counts[1]])
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        
    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        return MLPNet(input_dim=self.input_dim).to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    args = parser.parse_args()

    data_path = os.environ.get("data_file_path")
    scaler_path = os.environ.get("scaler_save_path")

    print(f"Using data path: {data_path}")
    print(f"Using scaler path: {scaler_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    client = FraudClient(
        data_file_path=data_path,
        scaler_path=scaler_path,
        metrics=[BalancedAccuracy("balanced_accuracy"), F1("f1_score")],
        device=device,
        progress_bar = True
    )

    fl.client.start_client(server_address="0.0.0.0:1080", client=client.to_client())
    torch.cuda.empty_cache()
    client.shutdown()
