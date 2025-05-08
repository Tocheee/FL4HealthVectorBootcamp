import argparse
from functools import partial
from logging import INFO
from typing import Any
from pathlib import Path
import joblib

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager

from examples.models.mlp_classifier import MLPNet
from fl4health.reporting import JsonReporter, WandBReporter, WandBStepType
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.adaptive_constraint_servers.fedprox_server import FedProxServer
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import set_all_random_seeds

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule, AdaptiveConstraintServerCheckpointAndStateModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.servers.base_server import FlServer

def fit_config(
    batch_size: int,
    n_server_rounds: int,
    current_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "current_server_round": current_round,
    }

def get_input_dim_from_scaler(scaler_path: str) -> int:
    from fl4health.utils.load_data import TabularScaler
    import numpy as np
    import pandas as pd
    
    scaler: TabularScaler = joblib.load(scaler_path)
    # print("Numeric features:", scaler.numeric_features)
    # print("Categorical features:", scaler.categorical_features)

    # Create a dummy row with the same columns used during fitting
    feature_names = scaler.numeric_features + scaler.categorical_features
    dummy_data = pd.DataFrame(
        [[0.0 if col in scaler.numeric_features else "unknown" for col in feature_names]],
        columns=feature_names
    )

    transformed = scaler.transform(dummy_data)
    return transformed.shape[1]

def main(config: dict[str, Any], server_address: str, wandb_entity: str | None) -> None:
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    scaler_path = config["scaler_path"]  # <-- MUST be set in config.yaml
    input_dim = get_input_dim_from_scaler(scaler_path)
    

    model = MLPNet(input_dim=input_dim)

    # To facilitate checkpointing
    parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model.pkl"),
    ]
    checkpoint_and_state_module = AdaptiveConstraintServerCheckpointAndStateModule(
        model=model,  model_checkpointers=checkpointers
    )
    
    # Server performs simple FedAveraging as its server-side optimization strategy and potentially adapts the
    # FedProx proximal weight mu
    strategy = FedAvgWithAdaptiveConstraint(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
        adapt_loss_weight=config["adapt_proximal_weight"],
        initial_loss_weight=config["initial_proximal_weight"],
        loss_weight_delta=config["proximal_weight_delta"],
        loss_weight_patience=config["proximal_weight_patience"],
    )

    json_reporter = JsonReporter()
    client_manager = SimpleClientManager()
    reporters: list[BaseReporter] = [json_reporter]

    if wandb_entity:
        wandb_reporter = WandBReporter(
            WandBStepType.ROUND,
            project="FL4Health",  # Name of the project under which everything should be logged
            name="Server",  # Name of the run on the server-side
            group="FedProx Experiment",  # Group under which each of the FL run logging will be stored
            entity=wandb_entity,  # WandB user name
            tags=["Test", "FedProx"],
            job_type="server",
            notes="Testing WB reporting",
        )
        reporters.append(wandb_reporter)

    server = FedProxServer(
        client_manager=client_manager, 
        fl_config=config, 
        strategy=strategy, 
        reporters=reporters,
        accept_failures=False,
        checkpoint_and_state_module=checkpoint_and_state_module,
    )

    fl.server.start_server(
        server=server,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )
    # Shutdown the server gracefully
    server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/baf_fedprox_example/config.yaml",
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address to be used to communicate with the clients",
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

    config = load_config(args.config_path)
    log(INFO, f"Server Address: {args.server_address}")

    if args.wandb_entity:
        log(INFO, f"Weights and Biases Entity Provided: {args.wandb_entity}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    main(config, args.server_address, args.wandb_entity)
