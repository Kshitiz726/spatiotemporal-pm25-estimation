import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import xgboost as xgb
import matplotlib.pyplot as plt


# Load hourly scaled data with lags
X = np.load("X_scaled_kathmandu_hourly.npy")
y = np.load("y_scaled_kathmandu_hourly.npy")

# Chronological split
total_samples = X.shape[0]
train_end = int(total_samples * 0.7)
val_end = int(total_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Combine train + val for training XGBoost (optional)
X_trainval = np.vstack((X_train, X_val))
y_trainval = np.vstack((y_train, y_val))

# Initialize and train XGBoost regressor
model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model_xgb.fit(X_trainval, y_trainval.ravel(), verbose=False)

# Predict on test set
y_pred_scaled = model_xgb.predict(X_test).reshape(-1, 1)

# Load scaler for y to inverse transform
scaler_y = joblib.load("scaler_y_kathmandu_hourly.pkl")

# Inverse transform scaled predictions and true values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# Calculate evaluation metrics
r2 = r2_score(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mean_diff = (y_pred - y_true).mean()

print(f"XGBoost Hourly Results:")
print(f"R²: {r2:.4f}")
print(f"RMSE (µg/m³): {rmse:.4f}")
print(f"MAE (µg/m³): {mae:.4f}")
print(f"Mean Difference (µg/m³): {mean_diff:.4f}")

# Get feature importances
importances = model_xgb.feature_importances_

# Print top 5 most important features
sorted_idx = np.argsort(importances)[::-1]
print("\nTop 5 Feature Importances:")
for i in range(5):
    print(f"Feature {sorted_idx[i]}: {importances[sorted_idx[i]]:.4f}")

# Plot top 5 feature importances
plt.figure(figsize=(8, 5))
xgb.plot_importance(model_xgb, max_num_features=5, importance_type='gain', show_values=False)
plt.title("Top 5 Feature Importances (Gain)")
plt.tight_layout()
plt.show()
