import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from fl4health.utils.load_data import TabularScaler
import joblib
import os

def load_and_concat_all_variants(data_folder: Path) -> pd.DataFrame:
    # variant_files = list(data_folder.glob("Variant*.csv"))
    variant_files = list(data_folder.glob("base*.csv"))
    all_dfs = []

    for file in variant_files:
        df = pd.read_csv(file)

        # Drop irrelevant columns
        df = df.drop(columns=[
            "bank_months_count",
            "prev_address_months_count",
            "velocity_4w"
        ])

        # Handle missing values
        cols_missing = [
            'current_address_months_count',
            'session_length_in_minutes',
            'device_distinct_emails_8w',
            'intended_balcon_amount'
        ]
        df[cols_missing] = df[cols_missing].replace(-1, np.nan)
        df = df.dropna()

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def create_and_save_scaler(data_folder: str, scaler_save_path: str):
    data_folder = Path(data_folder)
    scaler_save_path = Path(scaler_save_path)

    full_df = load_and_concat_all_variants(data_folder)
    X = full_df.drop(columns=["fraud_bool"])

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    scaler = TabularScaler(numeric_cols, categorical_cols)
    scaler.fit_transform(X)  # Fit the internal scalers/encoders

    joblib.dump(scaler, scaler_save_path)
    print(f"Global scaler saved to {scaler_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True, help="Folder with Variant*.csv files")
    parser.add_argument("--scaler_save_path", type=str, required=True, help="Where to save the scaler (e.g. global_scaler.joblib)")
    args = parser.parse_args()

    create_and_save_scaler(args.data_folder, args.scaler_save_path)
