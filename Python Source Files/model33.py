import numpy as np
import joblib
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1️⃣ Load scaled data (Merra hourly)
X = np.load("X_scaled_kathmandu_hourly_merra.npy")
y = np.load("y_scaled_kathmandu_hourly_merra.npy")

# 2️⃣ Chronological train-val-test split (70%-15%-15%)
total_samples = X.shape[0]
train_end = int(total_samples * 0.7)
val_end = int(total_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")



# 3️⃣ Build fixed DNN model
def build_fixed_model():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
    model.add(BatchNormalization())
    
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    
    model.add(Dense(1, activation='linear'))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

model = build_fixed_model()

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)

# 4️⃣ Train model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 5️⃣ Save the model
model.save("dnn_pm25_kathmandu_hourly_merra_fixed.keras")
print("✅ Fixed DNN model saved as dnn_pm25_kathmandu_hourly_merra_fixed.keras")

# 6️⃣ Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# 7️⃣ Inverse scale predictions and true values for final metrics
scaler_y = joblib.load("scaler_y_kathmandu_hourly_merra.pkl")
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

r2 = r2_score(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mean_diff = (y_pred - y_true).mean()

print("✅ DNN Hourly Merra PM2.5 Results (Fixed Model):")
print(f"R² Score: {r2:.4f}")
print(f"RMSE (µg/m³): {rmse:.4f}")
print(f"MAE (µg/m³): {mae:.4f}")
print(f"Mean Difference (µg/m³): {mean_diff:.4f}")

# 8️⃣ Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss - DNN Hourly PM2.5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dnn_hourly_merra_training_validation_loss.png")
plt.show()
