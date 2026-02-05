import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# -----------------------
# 🔧 Configuration
# -----------------------
dataset_dir = "C:\\Users\\jahma\\Downloads\\CSV-01-12\\01-12"  # Adjust path if needed
model_output_path = os.path.join(dataset_dir, "xgboost_cicddos2019.model")

# -----------------------
# 🧠 Merge and Train on SYN, UDP, NTP
# -----------------------
selected_files = [
    "DrDoS_UDP.csv",
    "Syn.csv",
    "DrDoS_NTP.csv"
]
csv_files = [os.path.join(dataset_dir, f) for f in selected_files if os.path.exists(os.path.join(dataset_dir, f))]

print("🚀 Loading and combining selected attack files...")

merged_data = []
for file in csv_files:
    label_name = os.path.splitext(os.path.basename(file))[0]
    print(f"📄 Reading {os.path.basename(file)} with label '{label_name}'...")
    try:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()
        df['Label'] = label_name
        merged_data.append(df)
    except Exception as e:
        print(f"❌ Failed to read {file}: {e}")

if not merged_data:
    raise ValueError("No data loaded. Please check selected files and paths.")

# Merge all and clean
full_df = pd.concat(merged_data, ignore_index=True)
X = full_df.drop(columns=['Label'], errors='ignore')
y = full_df['Label']

# Filter numeric only
X = X.select_dtypes(include=['int64', 'float64'])
is_finite = np.isfinite(X).all(axis=1)
X = X[is_finite]
y = y[is_finite]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create DMatrix
params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'device': 'cuda',
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

print("🚀 Training model on combined SYN, UDP, and NTP attack data...")
dtrain = xgb.DMatrix(X, label=y_encoded)
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, "Train")], early_stopping_rounds=10)

# -----------------------
# 📌 Save model
# -----------------------
model.save_model(model_output_path)
print(f"✅ Model saved to {model_output_path}")
