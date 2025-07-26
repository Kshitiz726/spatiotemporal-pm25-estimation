import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# Load scaled data
X = np.load("X_scaled_kathmandu_hourly_merra.npy")
y = np.load("y_scaled_kathmandu_hourly_merra.npy").ravel()  # Flatten to 1D

# Load scalers
scaler_X = joblib.load("scaler_X_kathmandu_hourly_merra.pkl")
scaler_y = joblib.load("scaler_y_kathmandu_hourly_merra.pkl")

# Get feature names from scaler if available
try:
    feature_cols = list(scaler_X.feature_names_in_)
    print(f"Loaded {len(feature_cols)} feature names from scaler_X")
except AttributeError:
    print("scaler_X has no attribute 'feature_names_in_', reconstructing feature names manually")

    # Load the original dataframe to reconstruct feature names
    df = pd.read_csv("final_cleaned_dataset.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Cyclical features added manually (same as when preparing data)
    cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']

    # Columns excluded from features
    exclude_cols = ['datetime', 'pm25', 'hour', 'month'] + cyclical_features

    # Base weather columns
    weather_cols = df.columns.difference(exclude_cols).tolist()

    # Lag hours
    lag_hours = [1, 2, 3]

    feature_cols = []
    # Base features: weather + cyclical
    feature_cols.extend(weather_cols)
    feature_cols.extend(cyclical_features)

    # Lag features: pm25 lag + weather lags
    for lag in lag_hours:
        feature_cols.append(f'pm25_lag_{lag}')
        feature_cols.extend([f"{col}_lag_{lag}" for col in weather_cols])

    print(f"Manually reconstructed feature columns count: {len(feature_cols)}")

# Check feature length matches X shape
assert len(feature_cols) == X.shape[1], f"Feature count mismatch! {len(feature_cols)} vs {X.shape[1]}"

# Train-test split (80-20), no shuffle for time series
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# Define base model (without early stopping in RandomizedSearchCV)
base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Reduced hyperparameter grid to avoid too many combinations
param_dist = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 1, 2, 3, 4],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0]
}

# Setup RandomizedSearchCV with reduced n_iter to 15
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=15,
    scoring='r2',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)
print("Best hyperparameters found:")
print(random_search.best_params_)

# Train final model with best params (you can add early stopping here on test set)
best_params = random_search.best_params_
final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    **best_params
)

# Fit final model WITHOUT early stopping (avoid errors)
final_model.fit(X_train, y_train)

# Predict and inverse transform to original scale
y_pred_scaled = final_model.predict(X_test)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Calculate evaluation metrics
r2 = r2_score(y_test_orig, y_pred_orig)
mse = mean_squared_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
mean_diff = np.mean(y_pred_orig - y_test_orig)

print("\n✅ XGBoost Hourly PM2.5 Forecasting Results (After Tuning):")
print(f"R² Score: {r2:.4f}")
print(f"RMSE (µg/m³): {rmse:.4f}")
print(f"MAE  (µg/m³): {mae:.4f}")
print(f"Mean Difference (µg/m³): {mean_diff:.4f}")

# Get feature importances by gain
importances = final_model.get_booster().get_score(importance_type='gain')

# Sort features by importance
sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 Feature Importances by Gain:")
for i, (feat, score) in enumerate(sorted_features[:10]):
    print(f"{i+1}. {feat} : {score:.4f}")
