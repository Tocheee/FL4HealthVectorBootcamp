import argparse
import os
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.mlp_classifier import MLPNet
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.reporting import JsonReporter, WandBReporter, WandBStepType
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


from fl4health.utils.load_data import DataPrep
from fl4health.utils.metrics import Accuracy, BalancedAccuracy, F1, ROC_AUC, Precision, Recall
from fl4health.utils.class_weights import compute_class_counts

class FocalLoss(nn.Module):
    def __init__(self, alpha:float, gamma: float = 2.0, reduction: str = "mean"):
        """
        Focal Loss for binary classification.

        Args:
            alpha (float): Weighting factor for the rare class (positive class).
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets.float())
        probas = torch.sigmoid(inputs)
        pt = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class FraudFedProxClient(FedProxClient):
    def __init__(self, data_file_path: str, scaler_path: str, metrics: list, device: torch.device, reporters: list, progress_bar = True):
        super().__init__(data_path=data_file_path, metrics=metrics, device=device,  reporters = reporters, progress_bar = True)

        self.data_file_path = data_file_path
        self.scaler_path = scaler_path
        self.device = device
        self.metrics = metrics
        self.reporters = reporters

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
        # sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)
        # batch_size = narrow_dict_type(config, "batch_size", int)
        # train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)  
        
    def get_model(self, config: Config) -> nn.Module:
        return MLPNet(input_dim=self.input_dim).to(self.device)

    def get_criterion(self, config: Config) -> _Loss:
        class_counts = compute_class_counts(self.train_loader.dataset)
        print(class_counts)
        # Compute alpha: inverse frequency of positive class
        # If positives are rare (e.g., fraud cases), we want to upweight them.
        # This alpha = pos / (pos + neg) gives a fractional weight to the positive class.
        # Then, the loss uses alpha for the positive class, and 1 - alpha for the negative class.
        pos_count = class_counts.get(1, 1)
        neg_count = class_counts.get(0, 1)
        alpha = pos_count / (pos_count + neg_count)
        # print(f"Using dynamic alpha for FocalLoss: {alpha:.4f}")
        return FocalLoss(alpha=alpha, gamma=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8087",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    parser.add_argument(
        "--wandb_entity",
        action="store",
        type=str,
        help="Entity to be used for W and B logging. If not provided, then no W and B logging is performed.",
        required=False,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = os.environ.get("data_file_path")
    scaler_path = os.environ.get("scaler_save_path")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data_path = Path(args.dataset_path)
    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)
    # Get wandb_entity if provided
    wandb_entity = args.wandb_entity

    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    json_reporter = JsonReporter()
    reporters: list[BaseReporter] = [json_reporter]

    if wandb_entity:
        log(INFO, f"Weights and Biases Entity Provided: {wandb_entity}")
        # NOTE: name/id will be set automatically and are not initialized here.
        wandb_reporter = WandBReporter(
            WandBStepType.ROUND,
            project="FL4Health",  # Name of the project under which everything should be logged
            group="FedProx Experiment",  # Group under which each of the FL run logging will be stored
            entity=wandb_entity,  # WandB user name
            tags=["Test", "FedProx"],
            job_type="client",
            notes="Testing WB reporting",
        )
        reporters.append(wandb_reporter)

    client = FraudFedProxClient(
        data_file_path=data_path,
        scaler_path=scaler_path,
        metrics=[BalancedAccuracy("balanced_accuracy"),F1("f1_score"), ROC_AUC("roc_auc"), Precision("precision"), Recall("recall")],
        device=device,
        progress_bar = True, 
        reporters = reporters
    )
    
    fl.client.start_client(server_address=args.server_address, client=client.to_client())

    # Shutdown the client gracefully
    client.shutdown()
