#!/bin/bash

n_clients_to_start=2
config_path="examples/bank_account_fraud_example/config.yaml"
# dataset_path="/projects/federated_learning/RBC2/Bank_Account_Fraud/"
dataset_path="/projects/federated_learning/RBC2/share/"
server_output_file="examples/bank_account_fraud_example/server.out"
client_output_folder="examples/bank_account_fraud_example/"
scaler_save_path="examples/bank_account_fraud_example/global_scaler.joblib"  # Adjust as needed

# Step 1: Generate Global Scaler (run only once per session)
echo "Generating global scaler from data folder: ${dataset_path}"
python -m fl4health.utils.generate_global_scaler \
    --data_folder "${dataset_path}" \
    --scaler_save_path "${scaler_save_path}"

# Step 2: Start the server, divert the outputs to a server file
echo "Server logging at: ${server_output_file}"
nohup python -m examples.bank_account_fraud_example.server --config_path ${config_path} > ${server_output_file} 2>&1 &

# Sleep for 20 seconds to allow the server to come up.
sleep 20

# Step 3: Start a client for each Variant*.csv file
i=1
# for file in "${dataset_path}"/Variant*.csv; do
for file in "${dataset_path}"/base*.csv; do
    client_log_path="${client_output_folder}client_${i}.out"
    file_path=$(basename "$file")
    full_file_path="${dataset_path}${file_path}"
    
    echo "Starting Client $i with file: $full_file_path"
    data_file_path="$full_file_path" scaler_save_path="$scaler_save_path" nohup python -m examples.bank_account_fraud_example.client > "$client_log_path" 2>&1 &

    ((i++))
done
