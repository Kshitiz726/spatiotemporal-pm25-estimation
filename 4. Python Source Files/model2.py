import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import joblib
import matplotlib.pyplot as plt

# 1️⃣ Load scaled hourly data
X = np.load("X_scaled_kathmandu_hourly.npy")
y = np.load("y_scaled_kathmandu_hourly.npy")

# 2️⃣ Chronological split
total_samples = X.shape[0]
train_end = int(total_samples * 0.7)
val_end = int(total_samples * 0.85)
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 3️⃣ Build DNN 256-128-64-32 architecture
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.05),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

optimizer = AdamW(learning_rate=0.001, weight_decay=0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

# 4️⃣ Train
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, lr_schedule],
    verbose=1
)

# 5️⃣ Evaluate
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Save model
model.save("dnn_pm25_kathmandu_hourly.keras")
print("✅ Hourly DNN model saved as dnn_pm25_kathmandu_hourly.keras")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('DNN Hourly PM2.5 Training Loss')
plt.tight_layout()
plt.savefig("dnn_pm25_kathmandu_hourly_training_loss.png")
plt.show()

# Inverse transform for metric reporting
scaler_y = joblib.load("scaler_y_kathmandu_hourly.pkl")
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# Metrics
r2 = r2_score(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mean_diff = (y_pred - y_true).mean()

print(f"✅ DNN Hourly Results:")
print(f"R²: {r2:.4f}")
print(f"RMSE (µg/m³): {rmse:.4f}")
print(f"MAE (µg/m³): {mae:.4f}")
print(f"Mean Difference (µg/m³): {mean_diff:.4f}")
