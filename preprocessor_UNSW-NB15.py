import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -----------------------
# 📂 Paths
# -----------------------
raw_data_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\CSV Files\\Training and Testing Sets"
processed_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed"

# -----------------------
# 📥 Load UNSW Datasets
# -----------------------
train_unsw = pd.read_csv(os.path.join(raw_data_dir, "UNSW_NB15_training-set.csv"))
test_unsw = pd.read_csv(os.path.join(raw_data_dir, "UNSW_NB15_testing-set.csv"))

# -----------------------
# 🏷️ Map attack categories to CICIDS-style categories
# -----------------------
attack_mapping = {
    "Normal": 0,         # Benign
    "Generic": 1,        # DDoS
    "Fuzzers": 3,        # Brute Force
    "DoS": 1,            # DDoS
    "Reconnaissance": 2, # PortScan
    "Exploits": 5,       # Exploits
    "Backdoor": 4,       # Botnet
    "Analysis": 6,       # Web Attack
    "Shellcode": 5,      # Exploits
    "Worms": 4           # Botnet
}

# Apply attack mapping
train_unsw["attack_cat"] = train_unsw["attack_cat"].map(attack_mapping)
test_unsw["attack_cat"] = test_unsw["attack_cat"].map(attack_mapping)

# Drop rows where attack mapping failed
train_unsw.dropna(subset=["attack_cat"], inplace=True)
test_unsw.dropna(subset=["attack_cat"], inplace=True)

# Convert attack labels to integers
train_unsw["attack_cat"] = train_unsw["attack_cat"].astype(int)
test_unsw["attack_cat"] = test_unsw["attack_cat"].astype(int)

# -----------------------
# 🔖 Encode Labels (Ensure 15 Classes)
# -----------------------
cicids_label_list = np.arange(15)  # Ensure full range of 15 labels

label_encoder = LabelEncoder()
label_encoder.fit(cicids_label_list)  # Fit encoder with full label range

# 🔹 Store "Label" column **before dropping attack_cat**
train_unsw["Label"] = label_encoder.transform(train_unsw["attack_cat"])
test_unsw["Label"] = label_encoder.transform(test_unsw["attack_cat"])

# -----------------------
# 🧹 Drop Unnecessary Columns
# -----------------------
columns_to_drop = ["attack_cat", "label"]  # Remove binary label and raw attack name
train_unsw.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_unsw.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# -----------------------
# 🧠 Feature Mapping to CICIDS
# -----------------------
feature_mapping = {
    "dur": "Flow Duration",
    "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets",
    "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets",
    "rate": "Flow Bytes/s",
    "sinpkt": "Flow IAT Mean",
    "dinpkt": "Flow IAT Std",
    "tcprtt": "Fwd IAT Total",
    "synack": "Fwd IAT Mean",
    "ackdat": "Fwd IAT Std",
    "smean": "Average Packet Size",
    "dmean": "Average Packet Size",
    "proto": "Protocol",
    "stcpb": "Fwd TCP Base Sequence Number",
    "dtcpb": "Bwd TCP Base Sequence Number",
    "ct_state_ttl": "CT State TTL",
    "ct_srv_src": "CT Server Source",
    "ct_srv_dst": "CT Server Destination",
    "ct_dst_src_ltm": "CT Destination Source LTM",
    "is_ftp_login": "FTP Login Status",
    "ct_ftp_cmd": "FTP Command Count",
    "ct_flw_http_mthd": "HTTP Method Count",
    "ct_dst_ltm": "CT Destination LTM",
    "ct_src_dport_ltm": "CT Source DPort LTM",
    "ct_dst_sport_ltm": "CT Destination SPort LTM"
}

train_unsw.rename(columns=feature_mapping, inplace=True)
test_unsw.rename(columns=feature_mapping, inplace=True)

# -----------------------
# 💾 Extract Labels BEFORE any dropping
# -----------------------
y_train_unsw = train_unsw["Label"].copy()
y_test_unsw = test_unsw["Label"].copy()

# -----------------------
# ✅ Feature Alignment with CICIDS
# -----------------------
cicids_feature_list = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

# 🚨 Only drop extras AFTER saving the label
for col in cicids_feature_list:
    if col not in train_unsw.columns:
        print(f"⚠️ Adding missing feature: {col}")
        train_unsw[col] = 0
        test_unsw[col] = 0

# 🚀 Drop duplicate 'Average Packet Size' columns **while keeping one version**
duplicate_columns = [col for col in train_unsw.columns if "Average Packet Size" in col]

if len(duplicate_columns) > 1:
    print(f"⚠️ Removing duplicate columns: {duplicate_columns}")
    duplicate_columns.remove("Average Packet Size")  # Keep one version
    train_unsw.drop(columns=duplicate_columns, inplace=True)
    test_unsw.drop(columns=duplicate_columns, inplace=True)

# 🚀 Ensure 'Average Packet Size' exists before ordering
if "Average Packet Size" not in train_unsw.columns:
    print("⚠️ Restoring 'Average Packet Size' as it was removed incorrectly.")
    train_unsw["Average Packet Size"] = 0
    test_unsw["Average Packet Size"] = 0

extra = set(train_unsw.columns) - set(cicids_feature_list)
if extra:
    print(f"⚠️ Dropping extra columns: {extra}")
    train_unsw.drop(columns=extra, inplace=True)
    test_unsw.drop(columns=extra, inplace=True)

# -----------------------
# 💾 Save Final Data
# -----------------------
train_unsw.to_csv(os.path.join(processed_dir, "X_train_unsw_aligned.csv"), index=False)
test_unsw.to_csv(os.path.join(processed_dir, "X_test_unsw_aligned.csv"), index=False)
y_train_unsw.to_csv(os.path.join(processed_dir, "y_train_unsw_encoded.csv"), index=False)
y_test_unsw.to_csv(os.path.join(processed_dir, "y_test_unsw_encoded.csv"), index=False)

print("✅ UNSW features and labels aligned and saved!")