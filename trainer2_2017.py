import os
import xgboost as xgb
import pandas as pd

# -----------------------
# 📂 Paths
# -----------------------
dataset_dir = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\Processed"
model_path = os.path.join(dataset_dir, "xgboost_cicids.model")
save_path = os.path.join(dataset_dir, "xgboost_combined.model")

# -----------------------
# 📥 Load Processed UNSW Data
# -----------------------
X_train_unsw = pd.read_csv(os.path.join(dataset_dir, "X_train_unsw_aligned.csv"))
y_train_unsw = pd.read_csv(os.path.join(dataset_dir, "y_train_unsw_encoded.csv"))
X_test_unsw = pd.read_csv(os.path.join(dataset_dir, "X_test_unsw_aligned.csv"))
y_test_unsw = pd.read_csv(os.path.join(dataset_dir, "y_test_unsw_encoded.csv"))

# Ensure labels are correctly loaded as integers
y_train_unsw = y_train_unsw.values.ravel()
y_test_unsw = y_test_unsw.values.ravel()

# Load the original CICIDS feature order
X_train_cic = pd.read_csv(os.path.join(dataset_dir, "X_train_cic.csv"))
cicids_feature_order = X_train_cic.columns.tolist()

# Reorder UNSW features to match CICIDS order exactly
X_train_unsw = X_train_unsw[cicids_feature_order]
X_test_unsw = X_test_unsw[cicids_feature_order]

# Convert to XGBoost DMatrix
dtrain_unsw = xgb.DMatrix(X_train_unsw, label=y_train_unsw)
dtest_unsw = xgb.DMatrix(X_test_unsw, label=y_test_unsw)

# -----------------------
# 🚀 Load Pre-Trained Model
# -----------------------
print("🚀 Loading CICIDS-trained model and fine-tuning on UNSW-NB15...")
model = xgb.Booster()
model.load_model(model_path)

# -----------------------
# 🔧 XGBoost Parameters
# -----------------------
xgb_params = {
    'objective': 'multi:softprob',  # Multi-class classification
    'num_class': 15,  # Keep full class range from CICIDS
    'tree_method': 'hist',  # Faster training
    'device': 'cpu',  # Use CPU (adjust if using GPU)
    'eval_metric': 'mlogloss',  # Log loss for multi-class classification
    'learning_rate': 0.005,  # Lower LR to prevent catastrophic forgetting
    'max_depth': 6,  # Keep complexity manageable
    'subsample': 0.8,  # Sample data to avoid overfitting
    'colsample_bytree': 0.8,  # Sample features to avoid overfitting
    'random_state': 42
}

# -----------------------
# 🏋️ Fine-Tune Model with UNSW
# -----------------------
print("🚀 Fine-tuning on UNSW-NB15 dataset...")
model = xgb.train(
    xgb_params,
    dtrain_unsw,
    num_boost_round=200,  # Continue training with 200 more rounds
    evals=[(dtest_unsw, "Test")],
    xgb_model=model,  # Keep previous training weights
    early_stopping_rounds=20  # Stop if no improvement
)

# -----------------------
# 💾 Save Fine-Tuned Model
# -----------------------
model.save_model(save_path)
print(f"✅ Fine-tuned model saved as {save_path}!")