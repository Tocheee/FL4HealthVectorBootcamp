import argparse
from functools import partial
from typing import Any
from pathlib import Path
import joblib

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.models.mlp_classifier import MLPNet
from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters

def fit_config(
    batch_size: int,
    current_server_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "current_server_round": current_server_round,
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

def main(config: dict[str, Any]) -> None:#, , data_file_path: str
    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    scaler_path = config["scaler_path"]  # <-- MUST be set in config.yaml
    input_dim = get_input_dim_from_scaler(scaler_path)
    
    print(f"\n[Server] Input dimension set from global scaler: {input_dim}\n")
    model = MLPNet(input_dim = input_dim)

    # To facilitate checkpointing
    parameter_exchanger = FullParameterExchanger()
    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model.pkl"),
    ]
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=parameter_exchanger, model_checkpointers=checkpointers
    )

    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = FedAvg(
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
    )

    server = FlServer(
        client_manager=SimpleClientManager(),
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        accept_failures=False,
    )

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:1080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/bank_account_fraud_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    main(config)
