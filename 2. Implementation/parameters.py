import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1️⃣ Load scaled data
X = np.load("X_scaled_kathmandu.npy")
y = np.load("y_scaled_kathmandu.npy")

# 2️⃣ Load scalers (for inverse transform)
scaler_X = joblib.load("scaler_X_kathmandu.pkl")
scaler_y = joblib.load("scaler_y_kathmandu.pkl")

# 3️⃣ Split chronologically (70% train, 15% val, 15% test)
total_samples = X.shape[0]
train_end = int(total_samples * 0.7)
val_end = int(total_samples * 0.85)

X_test = X[val_end:]
y_test = y[val_end:]

# 4️⃣ Load model with compile=False and recompile
model = load_model("dnn_pm25_kathmandu.h5", compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 5️⃣ Predict on test set
y_pred_scaled = model.predict(X_test)

# 6️⃣ Inverse transform predictions and true values to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# 7️⃣ Calculate metrics
r2 = r2_score(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
mean_diff = (y_pred - y_true).mean()

# 8️⃣ Print results
print(f"R² Score: {r2:.4f}")
print(f"RMSE (µg/m³): {rmse:.4f}")
print(f"MAE (µg/m³): {mae:.4f}")
print(f"Mean Difference (µg/m³): {mean_diff:.4f}")
