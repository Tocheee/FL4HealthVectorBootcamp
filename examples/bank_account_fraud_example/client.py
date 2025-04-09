import argparse
from pathlib import Path

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
from fl4health.utils.load_data import load_data
from fl4health.utils.metrics import Accuracy, BalancedAccuracy, F1

from fl4health.utils.class_weights import compute_class_counts

class FraudClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_data(self.data_path, batch_size)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        return None
    
    def get_criterion(self, config: Config) -> _Loss:
        counts = compute_class_counts(self.data_path)
        pos_weight = torch.tensor([counts[0] / counts[1]])  # more fraud penalty
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        input_dim = 68  # Adjust based on actual processed feature size
        return MLPNet(input_dim = input_dim).to(self.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    print('\n', data_path)
    client = FraudClient(data_path, [BalancedAccuracy("balanced_accuracy"), F1("f1_score")], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
