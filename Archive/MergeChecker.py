import pandas as pd
import os

dataset_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed"

# Load CICIDS Data
cic_x_train = pd.read_csv(f"{dataset_dir}\\X_train.csv")
cic_y_train = pd.read_csv(f"{dataset_dir}\\y_train.csv")

# Load UNSW-NB15 Data
unsw_x_train = pd.read_csv(f"{dataset_dir}\\Set 2\\X_train_unsw.csv")
unsw_y_train = pd.read_csv(f"{dataset_dir}\\Set 2\\y_train_unsw.csv")

print("🔍 Data Size Before Merging:")
print(f"CICIDS X_train: {cic_x_train.shape}, Y_train: {cic_y_train.shape}")
print(f"UNSW  X_train: {unsw_x_train.shape}, Y_train: {unsw_y_train.shape}")