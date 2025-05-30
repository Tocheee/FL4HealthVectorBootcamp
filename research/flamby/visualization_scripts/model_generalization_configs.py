# File name mapped to tuples of name appearing on the graph, variable name for array, keys for the mean
fed_isic_file_names_to_info: list[tuple[str, str, str, list[str]]] = [
    (
        "client_0_eval_performance.txt",
        "Local 0",
        "local0",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "client_1_eval_performance.txt",
        "Local 1",
        "local1",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "client_2_eval_performance.txt",
        "Local 2",
        "local2",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "client_3_eval_performance.txt",
        "Local 3",
        "local3",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "client_4_eval_performance.txt",
        "Local 4",
        "local4",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "client_5_eval_performance.txt",
        "Local 5",
        "local5",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "central_eval_performance.txt",
        "Central",
        "central",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "fedavg_eval_performance.txt",
        "FedAvg",
        "fedavg",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "fedavg_eval_performance.txt",
        "FedAvg_L",
        "fedavg_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
            "client_4_model_local_avg",
            "client_5_model_local_avg",
        ],
    ),
    (
        "fedadam_eval_performance.txt",
        "FedAdam",
        "fedadam",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "fedadam_eval_performance.txt",
        "FedAdam_L",
        "fedadam_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
            "client_4_model_local_avg",
            "client_5_model_local_avg",
        ],
    ),
    (
        "fedprox_eval_performance.txt",
        "FedProx",
        "fedprox",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "fedprox_eval_performance.txt",
        "FedProx_L",
        "fedprox_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
            "client_4_model_local_avg",
            "client_5_model_local_avg",
        ],
    ),
    (
        "scaffold_eval_performance.txt",
        "SCAFFOLD",
        "scaffold",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
            "server_model_client_4_avg",
            "server_model_client_5_avg",
        ],
    ),
    (
        "scaffold_eval_performance.txt",
        "SCAFFOLD_L",
        "scaffold_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
            "client_4_model_local_avg",
            "client_5_model_local_avg",
        ],
    ),
    (
        "fenda_eval_performance_001.txt",
        "FENDA",
        "fenda",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
            "client_4_model_local_avg",
            "client_5_model_local_avg",
        ],
    ),
    (
        "apfl_eval_performance.txt",
        "APFL",
        "apfl",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
            "client_4_model_local_avg",
            "client_5_model_local_avg",
        ],
    ),
]

# File name mapped to tuples of name appearing on the graph, variable name for array
fed_heart_disease_file_names_to_info: list[tuple[str, str, str, list[str]]] = [
    (
        "client_0_eval_performance_small_model.txt",
        "Local 0_S",
        "local0s",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "client_0_eval_performance_big_model.txt",
        "Local 0_L",
        "local0l",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "client_1_eval_performance_small_model.txt",
        "Local 1_S",
        "local1s",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "client_1_eval_performance_big_model.txt",
        "Local 1_L",
        "local1l",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "client_2_eval_performance_small_model.txt",
        "Local 2_S",
        "local2s",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "client_2_eval_performance_big_model.txt",
        "Local 2_L",
        "local2l",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "client_3_eval_performance_small_model.txt",
        "Local 3_S",
        "local3s",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "client_3_eval_performance_big_model.txt",
        "Local 3_L",
        "local3l",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "central_eval_performance_small_model.txt",
        "Central_S",
        "centrals",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "central_eval_performance_big_model.txt",
        "Central_L",
        "centrall",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "fedavg_eval_performance_small_model.txt",
        "FedAvg_{SS}",
        "fedavgs",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "fedavg_eval_performance_small_model.txt",
        "FedAvg_{LS}",
        "fedavg_locals",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "fedavg_eval_performance_big_model.txt",
        "FedAvg_{SL}",
        "fedavgl",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "fedavg_eval_performance_big_model.txt",
        "FedAvg_{LL}",
        "fedavg_locall",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "fedadam_eval_performance_small_model.txt",
        "FedAdam_{SS}",
        "fedadams",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "fedadam_eval_performance_small_model.txt",
        "FedAdam_{LS}",
        "fedadam_locals",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "fedadam_eval_performance_big_model.txt",
        "FedAdam_{SL}",
        "fedadaml",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "fedadam_eval_performance_big_model.txt",
        "FedAdam_{LL}",
        "fedadam_locall",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "fedprox_eval_performance_small_model.txt",
        "FedProx_{SS}",
        "fedproxss",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "fedprox_eval_performance_small_model.txt",
        "FedProx_{LS}",
        "fedprox_locals",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "fedprox_eval_performance_big_model.txt",
        "FedProx_{SL}",
        "fedproxl",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "fedprox_eval_performance_big_model.txt",
        "FedProx_{LL}",
        "fedprox_locall",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "scaffold_eval_performance_small_model.txt",
        "SCAFFOLD_{SS}",
        "scaffoldss",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "scaffold_eval_performance_small_model.txt",
        "SCAFFOLD_{LS}",
        "scaffold_locals",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "scaffold_eval_performance_big_model.txt",
        "SCAFFOLD_{SL}",
        "scaffoldl",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
            "server_model_client_3_avg",
        ],
    ),
    (
        "scaffold_eval_performance_big_model.txt",
        "SCAFFOLD_{LL}",
        "scaffold_locall",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "fenda_eval_performance_big_model.txt",
        "FENDA",
        "fenda",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
    (
        "apfl_eval_performance_big_model.txt",
        "APFL",
        "apfl",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
            "client_3_model_local_avg",
        ],
    ),
]

# File name mapped to tuples of name appearing on the graph, variable name for array, keys for the mean
fed_ixi_file_names_to_info: list[tuple[str, str, str, list[str]]] = [
    (
        "client_0_eval_performance.txt",
        "Local 0",
        "local0",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "client_1_eval_performance.txt",
        "Local 1",
        "local1",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "client_2_eval_performance.txt",
        "Local 2",
        "local2",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "central_eval_performance.txt",
        "Central",
        "central",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "fedavg_eval_performance.txt",
        "FedAvg",
        "fedavg",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "fedavg_eval_performance.txt",
        "FedAvg_L",
        "fedavg_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
        ],
    ),
    (
        "fedadam_eval_performance.txt",
        "FedAdam",
        "fedadam",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "fedadam_eval_performance.txt",
        "FedAdam_L",
        "fedadam_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
        ],
    ),
    (
        "fedprox_eval_performance.txt",
        "FedProx",
        "fedprox",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "fedprox_eval_performance.txt",
        "FedProx_L",
        "fedprox_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
        ],
    ),
    (
        "scaffold_eval_performance.txt",
        "SCAFFOLD",
        "scaffold",
        [
            "server_model_client_0_avg",
            "server_model_client_1_avg",
            "server_model_client_2_avg",
        ],
    ),
    (
        "scaffold_eval_performance.txt",
        "SCAFFOLD_L",
        "scaffold_local",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
        ],
    ),
    (
        "fenda_eval_performance.txt",
        "FENDA",
        "fenda",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
        ],
    ),
    (
        "apfl_eval_performance.txt",
        "APFL",
        "apfl",
        [
            "client_0_model_local_avg",
            "client_1_model_local_avg",
            "client_2_model_local_avg",
        ],
    ),
]
