import os
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Load Processed Data
# -----------------------
dataset_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed"

X_train_cic = pd.read_csv(os.path.join(dataset_dir, "X_train_cic.csv"))
y_train_cic = pd.read_csv(os.path.join(dataset_dir, "y_train_cic.csv"))
X_test_cic = pd.read_csv(os.path.join(dataset_dir, "X_test_cic.csv"))
y_test_cic = pd.read_csv(os.path.join(dataset_dir, "y_test_cic.csv"))

# Encode Labels
label_encoder = LabelEncoder()
y_train_cic = label_encoder.fit_transform(y_train_cic["Label"].values.ravel())
y_test_cic = label_encoder.transform(y_test_cic["Label"].values.ravel())

# Convert to DMatrix
dtrain_cic = xgb.DMatrix(X_train_cic, label=y_train_cic)
dtest_cic = xgb.DMatrix(X_test_cic, label=y_test_cic)

# XGBoost Parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),  # Ensure correct number of attack classes
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Train model on CICIDS
print("🚀 Training on CICIDS dataset...")
model = xgb.train(
    xgb_params,
    dtrain_cic,
    num_boost_round=100,
    evals=[(dtest_cic, "Test")],
    early_stopping_rounds=10
)

# Save Model
model.save_model(os.path.join(dataset_dir, "xgboost_cicids.model"))
print("✅ Model trained on CICIDS and saved.")